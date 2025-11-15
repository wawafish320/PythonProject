// Fill out your copyright notice in the Description page of Project Settings.


#include "Public/Anim/EnemyAnimInstance.h"
#include "Animation/AnimTypes.h"  
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNETypes.h"
#include "onnxruntime_cxx_api.h"
#endif
#include "Animation/AnimInstanceProxy.h"
#include "Components/CapsuleComponent.h"
#include "Templates/Atomic.h"
#include "Algo/Sort.h"

#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"




// 简明统计
struct FStats { float Min=0, Max=0, MeanAbs=0, L2=0; };

namespace
{
inline float* PtrOrNull(TArray<float>& Array)
{
	return Array.Num() > 0 ? Array.GetData() : nullptr;
}

inline const float* PtrOrNull(const TArray<float>& Array)
{
	return Array.Num() > 0 ? Array.GetData() : nullptr;
}

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
static int64 CountTensorElements(const UE::NNE::FTensorDesc& Desc)
{
	int64 Count = 1;
	const TConstArrayView<int32> Shape = Desc.GetShape().GetData();
	if (Shape.Num() == 0)
	{
		return 0;
	}
	for (const int32 Dim : Shape)
	{
		if (Dim <= 0)
		{
			return 0;
		}
		Count *= Dim;
	}
	return Count;
}
#endif

static float ComputePercentile(TArray<float>& Samples, float Percent)
{
	if (Samples.IsEmpty())
	{
		return 0.f;
	}
	Percent = FMath::Clamp(Percent, 0.f, 100.f);
	Algo::Sort(Samples);
	const float Pos = (Percent / 100.f) * (Samples.Num() - 1);
	const int32 Idx = FMath::Clamp(FMath::FloorToInt(Pos), 0, Samples.Num() - 1);
	const int32 Idx1 = FMath::Clamp(Idx + 1, 0, Samples.Num() - 1);
	const float Alpha = Pos - (float)Idx;
	return FMath::Lerp(Samples[Idx], Samples[Idx1], Alpha);
}

static float ClampStd(float Value)
{
	return FMath::Max(Value, 1e-4f);
}

static FORCEINLINE void DecodeRot6DToMatrix(const float* Data, FMatrix& OutMatrix)
{
	FVector X(Data[0], Data[1], Data[2]);
	FVector Z(Data[3], Data[4], Data[5]);
	if (X.IsNearlyZero())
	{
		X = FVector::ForwardVector;
	}
	else
	{
		X.Normalize();
	}
	Z = Z - FVector::DotProduct(Z, X) * X;
	if (Z.IsNearlyZero())
	{
		Z = FVector::UpVector;
	}
	else
	{
		Z.Normalize();
	}
	FVector Y = FVector::CrossProduct(Z, X).GetSafeNormal();
	if (FVector::DotProduct(X, FVector::CrossProduct(Y, Z)) < 0.f)
	{
		Y *= -1.f;
	}
	Z = FVector::CrossProduct(X, Y).GetSafeNormal();
	OutMatrix = FMatrix::Identity;
	OutMatrix.SetAxis(0, X);
	OutMatrix.SetAxis(1, Y);
	OutMatrix.SetAxis(2, Z);
}

static FORCEINLINE void EncodeMatrixToRot6D(const FMatrix& Matrix, float* OutData)
{
	const FVector X = Matrix.GetScaledAxis(EAxis::X).GetSafeNormal();
	const FVector Z = Matrix.GetScaledAxis(EAxis::Z).GetSafeNormal();
	OutData[0] = X.X; OutData[1] = X.Y; OutData[2] = X.Z;
	OutData[3] = Z.X; OutData[4] = Z.Y; OutData[5] = Z.Z;
}
}

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
class FFusedEventMotionModel
{
public:
	bool Initialize(UNNEModelData* ModelAsset, const FFusedModelInputDims& InputDims)
	{
		if (!ModelAsset)
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] Model asset missing."));
			return false;
		}

		FString CpuRuntimeName;
		TArray<FString> CpuNames = UE::NNE::GetAllRuntimeNames();
		for (const FString& Name : CpuNames)
		{
			if (Name == TEXT("NNERuntimeORTCpu"))
			{
				CpuRuntimeName = Name;
				break;
			}
		}
		if (CpuRuntimeName.IsEmpty() && CpuNames.Num() > 0)
		{
			CpuRuntimeName = CpuNames[0];
		}

		if (CpuRuntimeName.IsEmpty())
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] No CPU runtime detected."));
			return false;
		}

		TWeakInterfacePtr<INNERuntimeCPU> RuntimeCPU = UE::NNE::GetRuntime<INNERuntimeCPU>(CpuRuntimeName);
		if (!RuntimeCPU.IsValid())
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] Runtime %s not valid (plugin missing?)."), *CpuRuntimeName);
			return false;
		}

		Model = RuntimeCPU->CreateModelCPU(ModelAsset);
		if (!Model.IsValid())
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] CreateModelCPU failed."));
			return false;
		}

		ModelInstance = Model->CreateModelInstanceCPU();
		if (!ModelInstance.IsValid())
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] CreateModelInstanceCPU failed."));
			Model.Reset();
			return false;
		}

		if (!ConfigureInputShapes(InputDims))
		{
			ModelInstance.Reset();
			Model.Reset();
			return false;
		}

		OutputScratch.Reset();
		return true;
	}

	bool IsReady() const
	{
		return ModelInstance.IsValid();
	}

	bool Forward(const TArray<float>& State,
	             const TArray<float>& Cond,
	             const TArray<float>& Contacts,
	             const TArray<float>& AngVel,
	             const TArray<float>& PoseHist,
	             TArray<float>& OutMotion)
	{
		using namespace UE::NNE;

	if (!ModelInstance.IsValid())
	{
		return false;
	}

		const TConstArrayView<FTensorDesc> InputDescs  = ModelInstance->GetInputTensorDescs();
		const TConstArrayView<FTensorDesc> OutputDescs = ModelInstance->GetOutputTensorDescs();
		if (InputDescs.Num() < 5 || OutputDescs.Num() < 1)
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] Unexpected tensor layout. Inputs=%d Outputs=%d"), InputDescs.Num(), OutputDescs.Num());
			return false;
		}

		if (InputDescs.Num() != 5)
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] Expected exactly 5 inputs (state/cond/contacts/angvel/pose_hist). Got %d."), InputDescs.Num());
			return false;
		}

		TArray<FTensorBindingCPU> Inputs;
		Inputs.SetNum(InputDescs.Num());
		auto MakeBinding = [](const TArray<float>& Buffer)->FTensorBindingCPU
		{
			float* Ptr = const_cast<float*>(PtrOrNull(Buffer));
			const SIZE_T Bytes = Buffer.Num() * sizeof(float);
			return FTensorBindingCPU(Ptr, Bytes);
		};
		Inputs[0] = MakeBinding(State);
		Inputs[1] = MakeBinding(Cond);
		Inputs[2] = MakeBinding(Contacts);
		Inputs[3] = MakeBinding(AngVel);
		Inputs[4] = MakeBinding(PoseHist);

		const int64 OutElems = CountTensorElements(OutputDescs[0]);
		if (OutElems <= 0)
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] Output tensor has invalid shape."));
			return false;
		}
		OutputScratch.SetNumUninitialized((int32)OutElems);

		TArray<FTensorBindingCPU> Outs;
		Outs.Add(FTensorBindingCPU(OutputScratch.GetData(), SIZE_T(OutputScratch.Num() * sizeof(float))));

		if (ModelInstance->RunSync(Inputs, Outs) != IModelInstanceCPU::ERunSyncStatus::Ok)
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] RunSync failed."));
			return false;
		}

		OutMotion = OutputScratch;
		return true;
	}

	const FFusedModelInputDims& GetBoundInputDims() const { return BoundInputDims; }

private:
	bool ConfigureInputShapes(const FFusedModelInputDims& InputDims)
	{
		using namespace UE::NNE;

		if (!ModelInstance.IsValid())
		{
			return false;
		}

		const TConstArrayView<FTensorDesc> InDescs = ModelInstance->GetInputTensorDescs();
		if (InDescs.Num() != 5)
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] Expected exactly 5 inputs (state/cond/contacts/angvel/pose_hist). Got %d."), InDescs.Num());
			return false;
		}

		auto DescribeSymbolicShape = [](const FTensorDesc& Desc)->FString
		{
			FString Result(TEXT("["));
			const TConstArrayView<int32> SymDims = Desc.GetShape().GetData();
			for (int32 i = 0; i < SymDims.Num(); ++i)
			{
				Result += FString::Printf(TEXT("%d"), SymDims[i]);
				if (i + 1 < SymDims.Num())
				{
					Result += TEXT(", ");
				}
			}
			Result += TEXT("]");
			return Result;
		};

		for (int32 i = 0; i < InDescs.Num(); ++i)
		{
			UE_LOG(LogTemp, Display, TEXT("[FusedModel] Input[%d] symbolic shape %s"), i, *DescribeSymbolicShape(InDescs[i]));
		}
		auto ExtractFeatureDim = [](const FTensorDesc& Desc)->int32
		{
			const TConstArrayView<int32> SymDims = Desc.GetShape().GetData();
			if (SymDims.Num() >= 2)
			{
				const int32 Feature = SymDims.Last();
				return FMath::Max(Feature, 1);
			}
			return 1;
		};

		FFusedModelInputDims FinalDims = InputDims;
		if (FinalDims.Motion <= 0)   FinalDims.Motion   = ExtractFeatureDim(InDescs[0]);
		if (FinalDims.Cond <= 0)     FinalDims.Cond     = ExtractFeatureDim(InDescs[1]);
		if (FinalDims.Contacts <= 0) FinalDims.Contacts = ExtractFeatureDim(InDescs[2]);
		if (FinalDims.AngVel <= 0)   FinalDims.AngVel   = ExtractFeatureDim(InDescs[3]);
		if (FinalDims.PoseHist <= 0) FinalDims.PoseHist = ExtractFeatureDim(InDescs[4]);

		UE_LOG(LogTemp, Display, TEXT("[FusedModel] Requested dims: motion=%d cond=%d contacts=%d angvel=%d pose_hist=%d"),
			FinalDims.Motion, FinalDims.Cond, FinalDims.Contacts, FinalDims.AngVel, FinalDims.PoseHist);

		auto Make2DShape = [](int32 FeatureDim)->FTensorShape
		{
			const uint32 SafeDim = static_cast<uint32>(FMath::Max(FeatureDim, 1));
			TArray<uint32, TInlineAllocator<2>> Ds;
			Ds.Add(1u); // batch size = 1
			Ds.Add(SafeDim);
			return FTensorShape::Make(Ds);
		};

		TArray<FTensorShape> Shapes;
		Shapes.SetNum(InDescs.Num());
		Shapes[0] = Make2DShape(FinalDims.Motion);
		Shapes[1] = Make2DShape(FinalDims.Cond);
		Shapes[2] = Make2DShape(FinalDims.Contacts);
		Shapes[3] = Make2DShape(FinalDims.AngVel);
		Shapes[4] = Make2DShape(FinalDims.PoseHist);

		const IModelInstanceCPU::ESetInputTensorShapesStatus Status = ModelInstance->SetInputTensorShapes(Shapes);
		if (Status != IModelInstanceCPU::ESetInputTensorShapesStatus::Ok)
		{
			UE_LOG(LogTemp, Error, TEXT("[FusedModel] SetInputTensorShapes failed (status=%d)."), static_cast<int32>(Status));
			return false;
		}

		BoundInputDims = FinalDims;
		return true;
	}

	TSharedPtr<UE::NNE::IModelCPU, ESPMode::ThreadSafe> Model;
	TSharedPtr<UE::NNE::IModelInstanceCPU, ESPMode::ThreadSafe> ModelInstance;
	TArray<float> OutputScratch;
	FFusedModelInputDims BoundInputDims;
};
#endif

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
void FFusedEventMotionModelDeleter::operator()(FFusedEventMotionModel* Ptr) const
{
	delete Ptr;
}
#endif
UEnemyAnimInstance::~UEnemyAnimInstance() = default;

// ---------------------- Proxy 生命周期 --------------------------------------
FAnimInstanceProxy* UEnemyAnimInstance::CreateAnimInstanceProxy()
{
    return new FEnemyAnimInstanceProxy(this);
}
void UEnemyAnimInstance::DestroyAnimInstanceProxy(FAnimInstanceProxy* InProxy)
{
    delete static_cast<FEnemyAnimInstanceProxy*>(InProxy);
}

// ---------------------- GT：每帧把 X 推给 Proxy（可选加噪声） ----------------
void UEnemyAnimInstance::PreUpdateAnimation(float DeltaSeconds)
{
	Super::PreUpdateAnimation(DeltaSeconds);

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
	// [止抖修正] 只有在首次启动或GT明确更新了状态后，才将X推送到Proxy
	// 这确立了GT作为状态X的唯一权威来源
	if (bIsFirstGTUpdate || bGTStateDirty)
	{
		// Log detailed info when pushing state
		UE_LOG(LogTemp, Warning, TEXT("[GT] Pushing state X to Proxy. Gen=%llu, IsFirstUpdate=%d, IsDirty=%d"),
			GT_StateGen,
			(int32)bIsFirstGTUpdate,
			(int32)bGTStateDirty);

		// 推到 Proxy（供 WT 使用）
		ProxyGT().PushFromGT(CurrentMotionStateNorm, ++GT_StateGen);
        
		// 推送后立即重置标记
		bGTStateDirty = false;
		bIsFirstGTUpdate = false;
	}
	else
	{
		// Log when state push is skipped
		UE_LOG(LogTemp, Verbose, TEXT("[GT] SKIPPED state X push. Gen=%llu"), GT_StateGen);
	}
#else
	// Teacher-only：无需推送 X 状态，PoseMailbox 直接在 Evaluate 中取用
#endif
}





void FEnemyAnimInstanceProxy::PushPoseLocalFromGT(const TArray<FTransform>& LocalTracked, const TArray<FName>& UseBones)
{
	FScopeLock Lock(&PayloadCS);
	Payload_LocalTracked = LocalTracked;
	Payload_BoneNames    = UseBones;
	bPayloadIsLocal      = true;
	bHasNewPayload       = true;
}


bool FEnemyAnimInstanceProxy::Evaluate(FPoseContext& Output)
{
	// UE_LOG(LogTemp, Log, TEXT("[Proxy::Eval] ENTER"));

    // === 读取 GT 推来的（局部）载荷 ===
    bool               bConsumeLocal = false;
    TArray<FTransform> LocalTracked;
    TArray<FName>      BoneNames;
    {
        FScopeLock Lock(&PayloadCS);
        if (bHasNewPayload && bPayloadIsLocal)
        {
            bConsumeLocal   = true;
            LocalTracked    = Payload_LocalTracked;
            BoneNames       = Payload_BoneNames;
            bHasNewPayload  = false;
        }
    }
  

    int32 NumWritten = 0;
    if (!bConsumeLocal)
    {
    	
        return false;
    }

	// 确认会写局部后再清空，避免“无载荷帧清成 T-Pose”
	Output.ResetToRefPose();

	

    const FBoneContainer& BC = Output.Pose.GetBoneContainer();
    const USkeleton* Skel = BC.GetSkeletonAsset();
    if (!Skel)
    {
        return false;
    }

    const FReferenceSkeleton& RefSkel = Skel->GetReferenceSkeleton();
    const int32 NumSk = RefSkel.GetNum();
    const int32 N     = FMath::Min(LocalTracked.Num(), BoneNames.Num());

    // --- Ref 本地、父表 ---
    TArray<FTransform> RefLocal;   RefLocal.SetNumUninitialized(NumSk);
    TArray<int32>      ParentSk;   ParentSk.SetNumUninitialized(NumSk);
    for (int32 sk=0; sk<NumSk; ++sk)
    {
        RefLocal[sk] = RefSkel.GetRefBonePose()[sk];
        ParentSk[sk] = RefSkel.GetParentIndex(sk);
    }

    // --- 深度（父亲在前，用来排序） ---
    TArray<int32> Depth; Depth.Init(-1, NumSk);
    for (int32 sk=0; sk<NumSk; ++sk)
    {
        if (Depth[sk] != -1) continue;
        TArray<int32> chain;
        int32 cur = sk;
        while (cur != INDEX_NONE && Depth[cur] == -1) { chain.Add(cur); cur = ParentSk[cur]; }
        int32 d = (cur==INDEX_NONE) ? 0 : Depth[cur] + 1;
        for (int32 i=chain.Num()-1; i>=0; --i) Depth[ chain[i] ] = d++;
    }

    // --- 跟踪骨 → Skeleton 索引；Skeleton 索引 → 跟踪骨 ---
    TArray<int32> SkOfTracked; SkOfTracked.SetNumUninitialized(N);
    TMap<int32,int32> TrackedBySk; TrackedBySk.Reserve(N);
    for (int32 i=0; i<N; ++i)
    {
        const int32 sk = RefSkel.FindBoneIndex(BoneNames[i]);
        SkOfTracked[i] = sk;
        if (sk != INDEX_NONE) TrackedBySk.Add(sk, i);
    }

    // 只保留能映射上的条目，并按深度“父先子后”
    TArray<int32> Order; Order.Reserve(N);
    for (int32 i=0; i<N; ++i) if (SkOfTracked[i]!=INDEX_NONE) Order.Add(i);
    Order.Sort([&](int32 A,int32 B){ return Depth[SkOfTracked[A]] < Depth[SkOfTracked[B]]; });

    // --- 1) 先把每个【跟踪骨】提升到“相对其最近被跟踪祖先”的组件空间 ---
    TArray<FTransform> CSTracked; CSTracked.SetNumZeroed(N);

    auto FindTrackedParentIndex = [&](int32 sk)->int32
    {
        // 返回：跟踪父的“跟踪数组索引”，若无则 -1
        int32 p = ParentSk[sk];
        while (p != INDEX_NONE)
        {
            if (const int32* iTP = TrackedBySk.Find(p)) return *iTP;
            p = ParentSk[p];
        }
        return -1;
    };

    for (int32 k=0; k<Order.Num(); ++k)
    {
        const int32 i  = Order[k];
        const int32 sk = SkOfTracked[i];

        // LocalFixed：用预测的旋转 + RefPose 的位移（保持骨长与形体）
        FTransform LocalFixed = LocalTracked[i];
        LocalFixed.SetTranslation(RefLocal[sk].GetTranslation());
        LocalFixed.NormalizeRotation();

        const int32 iTrackedParent = FindTrackedParentIndex(sk);
        if (iTrackedParent < 0)
        {
            // 没有被跟踪的祖先：把它视为“挂到组件空间原点”
            CSTracked[i] = LocalFixed;
        }
        else
        {
            CSTracked[i] = LocalFixed * CSTracked[iTrackedParent];
        }
    }

    // --- 2) 再把“相对跟踪父”的 CS 桥接到“相对真实父”的 Local，并写进 Pose ---
    for (int32 k=0; k<Order.Num(); ++k)
    {
        const int32 i  = Order[k];
        const int32 sk = SkOfTracked[i];

        const FCompactPoseBoneIndex CP = BC.GetCompactPoseIndexFromSkeletonIndex(sk);
        if (!CP.IsValid()) continue;

        const FTransform CS_child = CSTracked[i];

        // 组件空间的“真实父”变换
        FTransform CS_parent_actual = FTransform::Identity;

        const int32 iTrackedParent = FindTrackedParentIndex(sk);
        int32 skTrackedParent = (iTrackedParent>=0) ? SkOfTracked[iTrackedParent] : INDEX_NONE;

        if (iTrackedParent >= 0)
        {
            CS_parent_actual = CSTracked[iTrackedParent];

        	// 从“被跟踪父”一路乘 RefLocal 走到“真实父”
        	int32 cur = ParentSk[sk]; // 真实父
        	while (cur != skTrackedParent && cur != INDEX_NONE)
        	{
        		CS_parent_actual = RefLocal[cur] * CS_parent_actual;
        		cur = ParentSk[cur];
        	}            
        }
        else
        {
            // 该链条上没有任何被跟踪的祖先：仅用 RefLocal 累乘到根
            int32 cur = ParentSk[sk];
            while (cur != INDEX_NONE)
            {
                CS_parent_actual = RefLocal[cur] * CS_parent_actual;
                cur = ParentSk[cur];
            }
        }

    	FTransform Local = CS_child.GetRelativeTransform(CS_parent_actual);
    	Local.NormalizeRotation();


    	// --- 仅对“真正的 Skeleton 根”清零旋转（不要用拓扑序第一个骨！） ---
    	const bool bIsSkeletonRoot = (ParentSk[sk] == INDEX_NONE);
    	FTransform WrittenLocal = Local;
    	if (bIsSkeletonRoot)
    	{
    		WrittenLocal.SetRotation(FQuat::Identity);
    	}
    	Output.Pose[CP] = WrittenLocal;

    	// --- [DeltaUE] 写入与期望（Local）之间的角度差（跳过根骨） ---
#if !(UE_BUILD_SHIPPING)
    	if (!bIsSkeletonRoot)
    	{
    		const FQuat q_written = WrittenLocal.GetRotation();
    		const FQuat q_expect  = Local.GetRotation();
    		double dot = q_written | q_expect;
    		dot = FMath::Clamp(dot, -1.0, 1.0);
    		const double angDeg = FMath::RadiansToDegrees(2.0 * FMath::Acos(dot));
    		if (angDeg > 0.5) // 阈值可调
    		{
    			UE_LOG(LogTemp, Warning, TEXT("[DeltaUE] bone=%s Δ=%.3f° (written vs expect)"),
					*BoneNames[i].ToString(), angDeg);
    		}
    	}
#endif

    	++NumWritten;

    }

    UE_LOG(LogTemp, Warning, TEXT("[Proxy::Eval] LOCAL payload bridged to actual parents. Written=%d / Tracked=%d"),
           NumWritten, Order.Num());


    return NumWritten > 0;
}



// 依据 schema 指定的两列（xz / xy）把四元数编码回 6D
static void QuatTo6D_BySchema(const FQuat& Q, bool bXZ, FVector& C0, FVector& C1)
{
	const FQuatRotationMatrix R(Q);              // ✅ 正确的类型
	const FVector X = R.GetUnitAxis(EAxis::X);   // 旋转后的 X 轴
	const FVector Y = R.GetUnitAxis(EAxis::Y);   // 旋转后的 Y 轴
	const FVector Z = R.GetUnitAxis(EAxis::Z);   // 旋转后的 Z 轴

	C0 = X;                   // 第一列固定用 X
	C1 = bXZ ? Z : Y;         // 第二列按 schema：xz 或 xy
	C0.Normalize(); C1.Normalize();
}

static FQuat SixDToQuat_BySchema(const FVector& C0, const FVector& C1, bool bUseXZ)
{
	auto SafeAxis = [](FVector Axis, const FVector& Fallback)->FVector
	{
		FVector Out = Axis;
		if (!Out.Normalize() || Out.IsNearlyZero())
		{
			Out = Fallback;
		}
		return Out;
	};

	const FVector Axis0 = SafeAxis(C0, FVector::ForwardVector);
	FVector AxisAlt = C1 - FVector::DotProduct(C1, Axis0) * Axis0;
	AxisAlt = SafeAxis(AxisAlt, bUseXZ ? FVector::UpVector : FVector::RightVector);

	FVector AxisX = Axis0;
	FVector AxisY;
	FVector AxisZ;

	if (bUseXZ)
	{
		AxisY = SafeAxis(FVector::CrossProduct(AxisAlt, AxisX), FVector::RightVector);
		if (FVector::DotProduct(AxisX, FVector::CrossProduct(AxisY, AxisAlt)) < 0.f)
		{
			AxisY *= -1.f;
		}
		AxisZ = SafeAxis(FVector::CrossProduct(AxisX, AxisY), FVector::UpVector);
	}
	else
	{
		AxisY = AxisAlt;
		AxisZ = SafeAxis(FVector::CrossProduct(AxisX, AxisY), FVector::UpVector);
		if (FVector::DotProduct(AxisX, FVector::CrossProduct(AxisY, AxisZ)) < 0.f)
		{
			AxisZ *= -1.f;
		}
	}

	FMatrix M = FMatrix::Identity;
	M.SetAxis(0, AxisX);
	M.SetAxis(1, AxisY);
	M.SetAxis(2, AxisZ);
	FQuat Q(M);
	Q.Normalize();
	return Q;
}




void UEnemyAnimInstance::NativeInitializeAnimation()
{
	if (USkeletalMeshComponent* Skel = GetSkelMeshComponent())
	{
		Skel->VisibilityBasedAnimTickOption = EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;
		Skel->bEnableUpdateRateOptimizations = false;
	}
	if (!LoadSchemaAndStats())
	{
		UE_LOG(LogTemp, Error, TEXT("LoadSchemaAndStats failed. Aborting initialization."));
		return;
	}
	

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
	ResetARState();
#else
	UE_LOG(LogTemp, Log, TEXT("[AnimInit] ONNX runtime disabled — using teacher playback only."));
#endif

	if (bEnableTeacherPlayback)
	{
		if (!LoadTeacherClip())
		{
			UE_LOG(LogTemp, Warning, TEXT("[Teacher] Failed to load clip. Falling back to autoregressive mode."));
		}
	}
	else
	{
		TeacherFrames.Reset();
		bTeacherClipReady = false;
		TeacherFrameCursor = 0;
#if !ENEMY_ANIM_WITH_ONNX_RUNTIME
		// 没有模型可用时强制进入 Teacher 模式，保证仍然有输出
		bEnableTeacherPlayback = true;
		if (!LoadTeacherClip())
		{
			UE_LOG(LogTemp, Error, TEXT("[Teacher] Clip missing while model runtime is disabled."));
		}
#endif
	}

	AngVelPrevQ.Reset();

	

	RefCache = FRefCache{};
	FinalRotsCS.Reset();
	FinalPossCS.Reset();
	

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
	LoadOnnxModel();
	if (!NeuralNetwork.IsValid() || !NeuralNetwork->IsReady())
	{
		UE_LOG(LogTemp, Warning, TEXT("[AnimInit] Neural runtime unavailable. Remaining in teacher playback only."));
		bUseTeacherForcingEval = false;
	}
	else
	{
		CurrentMotionStateNorm.SetNumZeroed(InputDim);
		PredictedLocal.Init(FTransform::Identity, kTrackedBones.Num());

		bContactLeft  = true;
		bContactRight = true;

		ActionBits = FVector4(0,1,0,0);
	}
#endif
}

void UEnemyAnimInstance::NativeUpdateAnimation(float DeltaSeconds)
{
    constexpr bool LOG_STAGE = true;
    constexpr bool LOG_PROBE = true;

    Super::NativeUpdateAnimation(DeltaSeconds);

    const TArray<FName>& UseBones = kTrackedBones;
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
	const bool bModelReady = NeuralNetwork.IsValid() && NeuralNetwork->IsReady();
#else
	const bool bModelReady = false;
#endif
	const bool bTeacherActive = ShouldUseTeacherPlayback();
#if !ENEMY_ANIM_WITH_ONNX_RUNTIME
	if (!bTeacherActive)
	{
		UE_LOG(LogTemp, Warning, TEXT("[AnimUpdate] Teacher playback disabled but inference path stripped. No motion will be produced."));
		return;
	}
#endif
	const bool bTeacherFeedsModel = bTeacherActive && bUseTeacherForcingEval && bModelReady;
	if (!bTeacherFeedsModel)
	{
		bTeacherCondValid = false;
	}
    if ((!bTeacherActive && !bModelReady) || UseBones.Num() <= 0 || DeltaSeconds <= 0.f) return;

    auto SafeQ = [](FQuat q){ if (!q.IsNormalized() || q.ContainsNaN()) q.Normalize(); return q.ContainsNaN()? FQuat::Identity : q; };
    auto IsFiniteVec = [](const FVector& v){ return FMath::IsFinite(v.X) && FMath::IsFinite(v.Y) && FMath::IsFinite(v.Z); };

    // ===== 1) 固定步长 60Hz：每帧最多推进一次 =====
	const float TargetDt =
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
		(bTeacherActive ? FMath::Max(TeacherFrameDt, KINDA_SMALL_NUMBER) : (1.f / 60.f));
#else
		FMath::Max(TeacherFrameDt, KINDA_SMALL_NUMBER);
#endif
    AccumulatedTime += DeltaSeconds;
    AccumulatedTime = FMath::Min(AccumulatedTime, 2.f * TargetDt);

    bool advanced = false;
    if (AccumulatedTime + KINDA_SMALL_NUMBER >= TargetDt)
    {
        AccumulatedTime -= TargetDt;
		if (bTeacherActive)
		{
			const bool clipAdvanced = AdvanceTeacherClip(!bTeacherFeedsModel);
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
			if (bTeacherFeedsModel && clipAdvanced)
			{
				const bool stepOk = StepModel(TargetDt);
				if (!stepOk)
				{
					PublishTeacherFallbackPose();
				}
			}
#endif
			advanced = clipAdvanced;
		}
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
		else
		{
			StepModel(TargetDt);
			advanced = true;
		}
#endif
    }

    // ===== 2) 只尝试一次取样；失败就沿用上一帧 =====
    TArray<FTransform> LocalsSrc;
    TArray<float>      Y_Denorm;
    uint32             Seq = 0;

    static TArray<FTransform> sLastLocalUE; // 兜底缓存
    static uint32             sLastSeq = 0;

    const bool got = PoseBox.Consume(LocalsSrc, Y_Denorm, Seq, /*bLog=*/false);
    if (!got)
    {
        if (sLastLocalUE.Num() == UseBones.Num())
        {
            if (LOG_STAGE) UE_LOG(LogTemp, Display, TEXT("[Anim] Pose missing this frame — reuse last (Seq=%u)"), sLastSeq);
            ProxyGT().PushPoseLocalFromGT(sLastLocalUE, UseBones);
        }
        return;
    }
    if (LocalsSrc.Num() != UseBones.Num())
    {
        if (sLastLocalUE.Num() == UseBones.Num())
        {
            UE_LOG(LogTemp, Warning, TEXT("[Anim] LocalsSrc size mismatch (%d vs %d) — reuse last"), LocalsSrc.Num(), UseBones.Num());
            ProxyGT().PushPoseLocalFromGT(sLastLocalUE, UseBones);
        }
        return;
    }

    if (LOG_STAGE) UE_LOG(LogTemp, Display, TEXT("[Anim] Snapshot Seq=%u bones=%d"), Seq, LocalsSrc.Num());

    USkeletalMeshComponent* SKC  = GetSkelMeshComponent();
    const USkeletalMesh*    Mesh = SKC ? SKC->GetSkeletalMeshAsset() : nullptr;
    if (!Mesh) return;

    const int32 N = UseBones.Num();
    const FReferenceSkeleton& RS = Mesh->GetRefSkeleton();

    // ===== 2.5) 关键修复：确保 RefCache 与当前 Mesh/UseBones 对齐 =====
    auto RebuildRefCacheIfNeeded = [&]() -> bool
    {
        const bool need =
            (!RefCache.bOk) ||
            (RefCache.BoundMesh != Mesh) ||
            (RefCache.OffInParentLocal.Num() != N) ||
            (RefCache.TrackedParent.Num()   != N) ||
            (RefCache.TopoOrder.Num()       != N);

        if (!need) return true;

        RefCache.bOk = false;
        RefCache.BoundMesh = Mesh;
        RefCache.OffInParentLocal.SetNumZeroed(N);
        RefCache.TrackedParent.SetNumZeroed(N);
        RefCache.TopoOrder.SetNumUninitialized(N);

        // 1) 建父索引与局部平移
        for (int32 i=0; i<N; ++i)
        {
            const int32 sk = RS.FindBoneIndex(UseBones[i]);
            if (sk == INDEX_NONE)
            {
                UE_LOG(LogTemp, Error, TEXT("[RefCache] Bone %s not found in mesh."), *UseBones[i].ToString());
                return false;
            }

            const int32 skp = RS.GetParentIndex(sk);
            int32 trackedParent = -1;
            if (skp >= 0)
            {
                const FName parentName = RS.GetBoneName(skp);
                trackedParent = UseBones.IndexOfByKey(parentName);
            }
            RefCache.TrackedParent[i] = trackedParent;

            FVector off = RS.GetRefBonePose()[sk].GetTranslation();
            if (!IsFiniteVec(off) || off.SizeSquared() > 1e8f) off = FVector::ZeroVector; // 数值兜底
            RefCache.OffInParentLocal[i] = off;

            RefCache.TopoOrder[i] = i; // 初始假定已拓扑有序
        }

        // 2) 若 UseBones 不是父先于子的顺序，做一遍简单拓扑排序
        {
            TArray<int32> indeg; indeg.Init(0, N);
            for (int32 i=0;i<N;++i) { int p=RefCache.TrackedParent[i]; if (p>=0) ++indeg[i]; }

            // Kahn 简易版：父不算入度，直接将 parent<child 的现有顺序拎出
            TArray<int32> out; out.Reserve(N);
            TArray<bool> pushed; pushed.Init(false, N);

            // 先推所有 parent=-1 的根
            for (int32 i=0;i<N;++i) if (RefCache.TrackedParent[i] < 0) { out.Add(i); pushed[i]=true; }

            bool changed = true;
            while (out.Num() < N && changed)
            {
                changed = false;
                for (int32 i=0;i<N;++i)
                {
                    if (pushed[i]) continue;
                    const int32 p = RefCache.TrackedParent[i];
                    if (p<0 || pushed[p]) { out.Add(i); pushed[i]=true; changed=true; }
                }
            }
            // 若还有没推的，就按原顺序补齐
            for (int32 i=0;i<N;++i) if (!pushed[i]) out.Add(i);

            check(out.Num()==N);
            RefCache.TopoOrder = MoveTemp(out);
        }

        RefCache.bOk = true;
        UE_LOG(LogTemp, Display, TEXT("[RefCache] Rebuilt for mesh. N=%d"), N);
        return true;
    };

    if (!RebuildRefCacheIfNeeded())
    {
        UE_LOG(LogTemp, Error, TEXT("[Anim] RefCache not ready; skip this frame."));
        return;
    }

    // ===== 3) SRC -> UE 坐标系换基 =====
    PredictedLocal_UE.SetNum(N);
    const bool bApplyBindThisFrame = (bBindReady && !bSkipBindApplyOnce);
    static int sBindWarmupLeft = 0;

    for (int i = 0; i < N; ++i)
    {
        const FQuat Qsrc = SafeQ(LocalsSrc[i].GetRotation());
        FQuat Rue = bConjugateS2U ? SafeQ(CachedQ_SrcToUE * Qsrc * CachedQ_UEToSrc) : Qsrc;

        // if (bApplyBindThisFrame && BindUE.IsValidIndex(i)) { Rue = SafeQ(BindUE[i] * Rue); }
        PredictedLocal_UE[i] = FTransform(Rue, FVector::ZeroVector, FVector::OneVector);
    }
    if (bSkipBindApplyOnce) bSkipBindApplyOnce = false;
    if (sBindWarmupLeft > 0 && bApplyBindThisFrame) --sBindWarmupLeft;

    // ===== 4) 组装 LocalUE（带 RefPose 的平移 OffInParentLocal）=====
    TArray<FTransform> LocalUE; LocalUE.SetNum(N);
    for (int i = 0; i < N; ++i)
    {
        FVector Off = RefCache.OffInParentLocal.IsValidIndex(i) ? RefCache.OffInParentLocal[i] : FVector::ZeroVector;
        if (!IsFiniteVec(Off) || Off.SizeSquared() > 1e8f) Off = FVector::ZeroVector; // 再兜底一次
        LocalUE[i] = FTransform(PredictedLocal_UE[i].GetRotation(), Off, FVector::OneVector);
    }

    // ===== 5) （可选）推进世界根位移：仅当本帧推进过模型 =====
    if (advanced)
    {
        if (ACharacter* Ch = Cast<ACharacter>(GetOwningActor()))
        {
            static constexpr bool  kZeroZ        = true;
            static constexpr float kTestSpeedMps = 0.8f;          // 80 cm/s
            static constexpr float kMaxStepCm   = 80.f;

            const float stepLen_cm = FMath::Min(kTestSpeedMps * TargetDt * 100.f, kMaxStepCm);
            const FVector fwd      = Ch->GetActorForwardVector();
            const FVector dWorld   = fwd * stepLen_cm;
            Ch->AddActorWorldOffset(kZeroZ ? FVector(dWorld.X, dWorld.Y, 0.f) : dWorld, /*bSweep=*/true);
        }
    }

    // ===== 6) Debug 探针（FK）=====
    if (LOG_PROBE && RefCache.bOk && RefCache.TopoOrder.Num()==N)
    {
        FinalRotsCS.SetNum(N);
        FinalPossCS.SetNum(N);

        auto ComposeFK = [&](const FTransform& RootCS)
        {
            for (int32 pass=0; pass<N; ++pass)
            {
                const int32 i = RefCache.TopoOrder[pass];
                const int32 p = RefCache.TrackedParent.IsValidIndex(i) ? RefCache.TrackedParent[i] : -1;
                const FTransform CSparent = (p>=0)
                    ? FTransform(FinalRotsCS[p], FinalPossCS[p], FVector::OneVector)
                    : RootCS;
                const FTransform CS = LocalUE[i] * CSparent;
                FinalRotsCS[i] = CS.GetRotation();
                FinalPossCS[i] = CS.GetLocation();
            }
        };
        ComposeFK(FTransform::Identity);

        auto Probe = [&](const TCHAR* n)
        {
            const int32 idx = UseBones.IndexOfByKey(FName(n));
            if (idx >= 0)
            {
                const FVector& pos = FinalPossCS[idx];
                const FRotator rot = FinalRotsCS[idx].Rotator();
                UE_LOG(LogTemp, Warning, TEXT("[Probe-CS] %-10s P=(%.2f,%.2f,%.2f) R=(P=%.1f Y=%.1f R=%.1f)"),
                    n, pos.X, pos.Y, pos.Z, rot.Pitch, rot.Yaw, rot.Roll);
            }
        };
        Probe(TEXT("pelvis")); Probe(TEXT("spine_01")); Probe(TEXT("thigh_l")); Probe(TEXT("foot_l"));
    }

    // ===== 7) 推给 Proxy，并缓存 =====
    if (LOG_STAGE) UE_LOG(LogTemp, Display, TEXT("[Anim] Push to Proxy (N=%d, Seq=%u)"), N, Seq);
    ProxyGT().PushPoseLocalFromGT(LocalUE, UseBones);
    LastSeqApplied = Seq;

    sLastLocalUE = LocalUE;
    sLastSeq     = Seq;
}


void UEnemyAnimInstance::NativePostEvaluateAnimation()
{
	Super::NativePostEvaluateAnimation();
    
	USkeletalMeshComponent* Skel = GetSkelMeshComponent();
	const AActor* Owner = GetOwningActor();
	if (!Skel || !Owner) return;

	// 3.1 Mesh 组件的相对旋（仅观察）
	{
		const FRotator R = Skel->GetRelativeRotation();
		UE_LOG(LogTemp, Warning, TEXT("[MeshRel] %s relRot=%s"),
			*Skel->GetName(), *R.ToCompactString());
		// 常见 Yaw = -90°；这个 -90° 不应再被算进 RootDelta/条件向量
	}

	// 3.2 角色前向 vs 骨盆前向（组件空间），直行应≈0°
	const int32 PelvisIdx = Skel->GetBoneIndex(TEXT("pelvis"));   // 你的骨名里确有 "pelvis"
	if (PelvisIdx != INDEX_NONE)
	{
		// 组件空间 Transform
		const FTransform PelvisCS = Skel->GetBoneTransform(PelvisIdx);

		const FVector fActor  = Owner->GetActorForwardVector().GetSafeNormal2D();
		const FVector fPelvis = PelvisCS.GetRotation().RotateVector(FVector::ForwardVector).GetSafeNormal2D();

		const float cosA = FVector::DotProduct(fPelvis, fActor);
		const float deg  = FMath::RadiansToDegrees(FMath::Acos(FMath::Clamp(cosA, -1.f, 1.f)));
		float signedAngle = FMath::RadiansToDegrees(FMath::Atan2(fPelvis.X * fActor.Y - fPelvis.Y * fActor.X, cosA));
		signedAngle = -90.f;

		UE_LOG(LogTemp, Warning, TEXT("[FacingCheck] angle(ActorFwd, PelvisFwd)=%.1f deg  fActor=%s  fPelvis=%s"),
			deg, *V2_2D(fActor), *V2_2D(fPelvis));

		if (!bPelvisFacingAligned && FMath::Abs(signedAngle) > 1.f)
		{
			FRotator NewRel = Skel->GetRelativeRotation();
			NewRel.Yaw += signedAngle;
			Skel->SetRelativeRotation(NewRel);
			bPelvisFacingAligned = true;
			UE_LOG(LogTemp, Display, TEXT("[FacingCheck] Applied one-time mesh yaw correction %.1f deg to align pelvis."), signedAngle);
		}
	}
}

bool UEnemyAnimInstance::LoadSchemaAndStats()
{
    // --- 0) 打开主 JSON（单一入口） ---
    const FString SchemaPathAbs = FPaths::ConvertRelativePathToFull(SchemaJsonPath.FilePath);
    const FString SchemaDir     = FPaths::GetPath(SchemaPathAbs);
    UE_LOG(LogTemp, Log, TEXT("[Schema] %s"), *SchemaPathAbs);

    FString S;
    if (!FFileHelper::LoadFileToString(S, *SchemaPathAbs)) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] Cannot read file."));
        return false;
    }
    TSharedPtr<FJsonObject> Root;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(S), Root) || !Root.IsValid()) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] Parse failed."));
        return false;
    }
    const TSharedPtr<FJsonObject> Meta = Root->HasField(TEXT("meta")) ? Root->GetObjectField(TEXT("meta")) : nullptr;

    // --- 1) 基本维度 / fps ---
    InputDim  = Root->HasTypedField<EJson::Number>(TEXT("input_dim"))  ? Root->GetIntegerField(TEXT("input_dim"))  : 0;
    OutputDim = Root->HasTypedField<EJson::Number>(TEXT("output_dim")) ? Root->GetIntegerField(TEXT("output_dim")) : 0;
    if (Root->HasTypedField<EJson::Number>(TEXT("fps"))) {
        const double fps = Root->GetNumberField(TEXT("fps"));
        if (fps > 1e-3) { TrajHz = (float)fps; ModelDt = 1.f / (float)fps; }
        UE_LOG(LogTemp, Log, TEXT("[Schema] fps=%.3f -> ModelDt=%.6f"), fps, ModelDt);
    }

    // --- 2) 布局：只认 meta.state_layout / meta.output_layout（不再回退 feature_layout / 旧键名） ---
    auto ParseRangeLayout = [&](const TSharedPtr<FJsonObject>& Obj, const TCHAR* Key, TMap<FString,FStateSlice>& Out)->bool
    {
        Out.Reset();
        if (!Obj || !Obj->HasField(Key)) return false;
        const TSharedPtr<FJsonObject> L = Obj->GetObjectField(Key);
        for (const auto& KV : L->Values)
        {
            const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
            if (L->TryGetArrayField(KV.Key, Arr) && Arr && Arr->Num() >= 2)
            {
                const int32 a = (int32)(*Arr)[0]->AsNumber();
                const int32 b = (int32)(*Arr)[1]->AsNumber();
                FStateSlice SS; SS.Start = a; SS.Size = (b - a);
                Out.Add(KV.Key, SS);
            }
        }
        return Out.Num() > 0;
    };

    auto ParseStartSizeLayout = [&](const TSharedPtr<FJsonObject>& Obj, const TCHAR* Key, TMap<FString,FStateSlice>& Out)->bool
    {
        Out.Reset();
        if (!Obj || !Obj->HasTypedField<EJson::Object>(Key)) return false;

        TSharedPtr<FJsonObject> L = Obj->GetObjectField(Key);
        for (const auto& KV : L->Values)
        {
            TSharedPtr<FJsonObject> Item = KV.Value->AsObject();
            if (!Item.IsValid()) continue;

            double Start = 0, Size = 0;
            const bool ok1 = Item->TryGetNumberField(TEXT("start"), Start);
            const bool ok2 = Item->TryGetNumberField(TEXT("size"),  Size);
            if (!(ok1 && ok2)) continue;

            FStateSlice SS; SS.Start = (int32)Start; SS.Size = (int32)Size;
            Out.Add(KV.Key, SS);
        }
        return Out.Num() > 0;
    };

    // 只从 meta 读取
    bool bStateOK = ParseRangeLayout    (Meta, TEXT("state_layout"),  StateLayout)
                 || ParseStartSizeLayout(Meta, TEXT("state_layout"),  StateLayout);
    bool bOutOK   = ParseRangeLayout    (Meta, TEXT("output_layout"), OutputLayout)
                 || ParseStartSizeLayout(Meta, TEXT("output_layout"), OutputLayout);
    if (!bStateOK || !bOutOK) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] meta.state_layout / meta.output_layout missing or invalid."));
        return false;
    }

    // 若未提供 input_dim / output_dim，则由布局兜底
    if (InputDim  <= 0) { int32 maxe=0; for (auto& P:StateLayout)  { maxe=FMath::Max(maxe, P.Value.Start + P.Value.Size);}  InputDim  = maxe; }
    if (OutputDim <= 0) { int32 maxe=0; for (auto& P:OutputLayout) { maxe=FMath::Max(maxe, P.Value.Start + P.Value.Size);}  OutputDim = maxe; }

    // --- 2.1) 布局必备片段严格校验（缺失不回退） ---
    const FStateSlice* S_Yaw = StateLayout.Find(TEXT("RootYaw"));
    const FStateSlice* S_RV  = StateLayout.Find(TEXT("RootVelocity"));
    const FStateSlice* S_AV  = StateLayout.Find(TEXT("BoneAngularVelocities"));
    const FStateSlice* S_R6X = StateLayout.Find(TEXT("BoneRotations6D"));
    if (!S_Yaw || !S_RV || !S_AV || !S_R6X) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] state_layout missing required slice(s): RootYaw/RootVelocity/BoneAngularVelocities/BoneRotations6D."));
        return false;
    }
    const FStateSlice* S_R6Y = OutputLayout.Find(TEXT("BoneRotations6D"));
    if (!S_R6Y) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] output_layout missing required slice: BoneRotations6D."));
        return false;
    }
    if ((S_R6X->Size % 6) != 0 || (S_R6Y->Size % 6) != 0) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] Rot6D slice size must be multiple of 6 (X:%d, Y:%d)."), S_R6X->Size, S_R6Y->Size);
        return false;
    }

    // --- 3) 归一化统计（缺失不回退） ---
    MuX.Reset(); StdX.Reset(); MuY.Reset(); StdY.Reset(); DropY.Reset();
    auto ReadArray = [&](const TSharedPtr<FJsonObject>& O, const TCHAR* Key, TArray<float>& Out)->bool{
        const TArray<TSharedPtr<FJsonValue>>* A=nullptr;
        if (!O || !O->TryGetArrayField(Key, A) || !A) return false;
        Out.SetNum(A->Num());
        for (int32 i=0;i<A->Num();++i) Out[i]=(float)(*A)[i]->AsNumber();
        return true;
    };

    // Y 统计：必须
    if (!(ReadArray(Root, TEXT("MuY"), MuY) && ReadArray(Root, TEXT("StdY"), StdY))) {
        UE_LOG(LogTemp, Error, TEXT("[MuStd] MuY/StdY missing."));
        return false;
    }
    if (MuY.Num()!=OutputDim || StdY.Num()!=OutputDim){
        UE_LOG(LogTemp, Error, TEXT("[MuStd] Y dim mismatch (MuY:%d StdY:%d vs OutputDim:%d)."), MuY.Num(), StdY.Num(), OutputDim);
        return false;
    }

    // X 统计：必须
    if (!(ReadArray(Root, TEXT("MuX"), MuX) && ReadArray(Root, TEXT("StdX"), StdX))) {
        UE_LOG(LogTemp, Error, TEXT("[MuStd] MuX/StdX missing."));
        return false;
    }
    if (MuX.Num()!=InputDim || StdX.Num()!=InputDim){
        UE_LOG(LogTemp, Error, TEXT("[MuStd] X dim mismatch (MuX:%d StdX:%d vs InputDim:%d)."), MuX.Num(), StdX.Num(), InputDim);
        return false;
    }

    // --- 3.1) 读取 tanh scales（缺失/尺寸不匹配 -> 直接报错） ---
    TanhScalesRootVel.Reset();
    TanhScalesAngVel.Reset();
    if (!ReadArray(Root, TEXT("tanh_scales_rootvel"), TanhScalesRootVel)) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] tanh_scales_rootvel missing."));
        return false;
    }
    if (!ReadArray(Root, TEXT("tanh_scales_angvel"),  TanhScalesAngVel)) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] tanh_scales_angvel missing."));
        return false;
    }

	ReadArray(Root, TEXT("tanh_scales_pose_hist"), PoseHistoryScales);
	ReadArray(Root, TEXT("MuPoseHist"), PoseHistoryMu);
	ReadArray(Root, TEXT("StdPoseHist"), PoseHistoryStd);
	if (PoseHistoryScales.Num() > 0)
	{
		if (Root->HasTypedField<EJson::Number>(TEXT("pose_hist_len")))
		{
			PoseHistoryLen = Root->GetIntegerField(TEXT("pose_hist_len"));
		}
		if (PoseHistoryLen <= 0)
		{
			const int32 Bones = kTrackedBones.Num();
			if (Bones > 0)
			{
				const int32 Frames = PoseHistoryScales.Num() / (Bones * 6);
				PoseHistoryLen = FMath::Max(0, Frames);
			}
			else
			{
				PoseHistoryLen = 0;
			}
		}
	}
    if (TanhScalesRootVel.Num() != S_RV->Size) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] tanh_scales_rootvel length=%d mismatch RootVelocity.Size=%d."),
            TanhScalesRootVel.Num(), S_RV->Size);
        return false;
    }
    if (TanhScalesAngVel.Num() != S_AV->Size) {
        UE_LOG(LogTemp, Error, TEXT("[Schema] tanh_scales_angvel length=%d mismatch BoneAngularVelocities.Size=%d."),
            TanhScalesAngVel.Num(), S_AV->Size);
        return false;
    }

    // --- 4) Drop（初始化 + 6D 组内一致性保护；你的旧逻辑保留） ---
    DropY.Init(0, OutputDim);

#if !(UE_BUILD_SHIPPING)
    if (const FStateSlice* Rot = OutputLayout.Find(TEXT("BoneRotations6D")))
    {
        const int32 NumGroups = Rot->Size / 6;
        for (int32 g=0; g<NumGroups; ++g)
        {
            const int32 base = Rot->Start + g*6;
            int32 cntDrop = 0;
            for (int32 k=0;k<6;++k) if (DropY.IsValidIndex(base+k) && DropY[base+k]) ++cntDrop;
            if (cntDrop>0 && cntDrop<6) { for (int32 k=0;k<6;++k) DropY[base+k]=0; } // 统一改“保留整组”
        }
    }
#endif

	auto TryLoadCondNormJson = [&](const FString& Path)->bool
	{
		if (Path.IsEmpty() || !FPaths::FileExists(Path))
		{
			return false;
		}
		FString CondText;
		if (!FFileHelper::LoadFileToString(CondText, *Path))
		{
			return false;
		}
		TSharedPtr<FJsonObject> CondObj;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(CondText), CondObj) || !CondObj.IsValid())
		{
			return false;
		}

		
		return true;
	};
	

	CondNormMode = ECondNormMode::Raw;
	

    // --- 5) 结束 ---
    UE_LOG(LogTemp, Log, TEXT("[Schema OK] X=%d Y=%d C=%d fps=%.1f  (tanh_scales: rv=%d, ang=%d, pose_hist=%d | len=%d)"),
        InputDim, OutputDim, CondDim, TrajHz, TanhScalesRootVel.Num(), TanhScalesAngVel.Num(), PoseHistoryScales.Num(), PoseHistoryLen);
    return true;
}

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
void UEnemyAnimInstance::ResetARState()
{
	// 1) 前置校验
	if (MuX.Num() != InputDim || StdX.Num() != InputDim || StateLayout.IsEmpty())
	{
		UE_LOG(LogTemp, Error, TEXT("ResetARState failed: Schema/Stats not loaded correctly. Please check SchemaJsonPath."));
		return;
	}

	ResetCondHistory();
	CurrentTeacherContactsNorm.Reset();
	CurrentTeacherAngVelNorm.Reset();
	CurrentTeacherPoseHistNorm.Reset();
	bPelvisFacingAligned = false;

	// 1) 从 Seed JSON 取 raw，并且同时带出 UE-Local 的首帧姿态
	TArray<float>      X_raw;
	TArray<FTransform> SeedLocalUE;
	const bool bSeeded = TryBuildRawStateFromSeedJson(X_raw, &SeedLocalUE);
	if (!bSeeded)
	{
		UE_LOG(LogTemp, Warning, TEXT("[ResetAR] SeedJSON not used (missing/failed). Fallback to RefPose warm-start."));
		WarmStartPoseInRawState(X_raw);   // 你的兜底：用 RefPose 等方式构造 raw
		SeedLocalUE.Reset();              // 没首帧姿态就不推
	}

	// 2) （可选）重建条件向量，使起点与训练一致
	TArray<float> Cond7; Cond7.Init(0.f, 7);
	BuildCondVector(Cond7);

	// 3) 归一化 -> 当前帧 Z；Prev_X_raw 用该 RAW 初始化（对齐自回归历史）
	NormalizeXRaw_To_Z(X_raw, CurrentMotionStateNorm);
	Prev_X_raw = X_raw;

	// 4) 把首帧 UE-Local 姿态推给 Proxy，确保视觉姿态与 X_raw 首帧一致
	if (SeedLocalUE.Num() == kTrackedBones.Num())
	{
		GetProxyOnGameThread<FEnemyAnimInstanceProxy>().PushPoseLocalFromGT(SeedLocalUE, kTrackedBones);
		UE_LOG(LogTemp, Display, TEXT("[ResetAR] Seed pose pushed to Proxy (%d bones)"), SeedLocalUE.Num());

		TArray<FTransform> SeedLocalSrc;
		SeedLocalSrc.SetNum(SeedLocalUE.Num());
		for (int32 i = 0; i < SeedLocalUE.Num(); ++i)
		{
			FQuat Q = SeedLocalUE[i].GetRotation();
			if (bConjugateS2U)
			{
				Q = CachedQ_UEToSrc * Q * CachedQ_SrcToUE;
			}
			SeedLocalSrc[i].SetRotation(Q);
			SeedLocalSrc[i].SetTranslation(FVector::ZeroVector);
		}
		ResetPoseHistoryFromPose(SeedLocalSrc);
	}
	else
	{
		TArray<FTransform> IdentityPose;
		IdentityPose.Init(FTransform::Identity, kTrackedBones.Num());
		ResetPoseHistoryFromPose(IdentityPose);
	}

	UE_LOG(LogTemp, Log, TEXT("ResetARState: seed ready. caches cleared. (from %s)"),
		bSeeded ? TEXT("SeedJSON") : TEXT("RefPose+Cond"));
}
#endif

void UEnemyAnimInstance::NormalizeXRaw_To_Z(const TArray<float>& X_raw, TArray<float>& OutZ) const
{
	if (InputDim <= 0)
	{
		OutZ.Reset();
		return;
	}

	TArray<float> Prep;
	Prep.SetNum(InputDim);
	for (int32 i = 0; i < InputDim; ++i)
	{
		Prep[i] = (i < X_raw.Num()) ? X_raw[i] : 0.f;
	}

	const FStateSlice* sYaw = StateLayout.Find(TEXT("RootYaw"));
	const FStateSlice* sRV  = StateLayout.Find(TEXT("RootVelocity"));
	const FStateSlice* sAV  = StateLayout.Find(TEXT("BoneAngularVelocities"));

	if (sYaw)
	{
		for (int32 k = 0; k < sYaw->Size; ++k)
		{
			const int32 idx = sYaw->Start + k;
			const float RawYaw = (idx < X_raw.Num()) ? X_raw[idx] : 0.f;
			const float Wrapped = FMath::UnwindRadians(RawYaw);
			Prep[idx] = Wrapped / PI;
		}
	}

	if (sRV)
	{
		check(TanhScalesRootVel.Num() == sRV->Size);
		for (int32 k = 0; k < sRV->Size; ++k)
		{
			const int32 idx = sRV->Start + k;
			const float Scale = FMath::Max(FMath::Abs(TanhScalesRootVel[k]), 1e-3f);
			const float Raw = (idx < X_raw.Num()) ? X_raw[idx] : 0.f;
			Prep[idx] = FMath::Tanh(Raw / Scale);
		}
	}

	if (sAV)
	{
		check(TanhScalesAngVel.Num() == sAV->Size);
		for (int32 k = 0; k < sAV->Size; ++k)
		{
			const int32 idx = sAV->Start + k;
			const float Scale = FMath::Max(FMath::Abs(TanhScalesAngVel[k]), 1e-3f);
			const float Raw = (idx < X_raw.Num()) ? X_raw[idx] : 0.f;
			Prep[idx] = FMath::Tanh(Raw / Scale);
		}
	}

	OutZ.SetNum(InputDim);
	for (int32 i = 0; i < InputDim; ++i)
	{
		const float mu = MuX.IsValidIndex(i)  ? MuX[i]  : 0.f;
		const float sd = StdX.IsValidIndex(i) ? FMath::Max(FMath::Abs(StdX[i]), 1e-6f) : 1.f;
		const float v  = Prep[i];
		OutZ[i] = (v - mu) / sd;
	}
}

void UEnemyAnimInstance::DenormX_Z_To_Raw(const TArray<float>& X_norm, TArray<float>& OutRaw) const
{
	if (InputDim <= 0)
	{
		OutRaw.Reset();
		return;
	}

	OutRaw.SetNum(InputDim);
	for (int32 i = 0; i < InputDim; ++i)
	{
		const float mu = MuX.IsValidIndex(i)  ? MuX[i]  : 0.f;
		const float sd = StdX.IsValidIndex(i) ? FMath::Max(FMath::Abs(StdX[i]), 1e-6f) : 1.f;
		const float zn = X_norm.IsValidIndex(i) ? X_norm[i] : 0.f;
		OutRaw[i] = zn * sd + mu;
	}

	const FStateSlice* sYaw = StateLayout.Find(TEXT("RootYaw"));
	const FStateSlice* sRV  = StateLayout.Find(TEXT("RootVelocity"));
	const FStateSlice* sAV  = StateLayout.Find(TEXT("BoneAngularVelocities"));

	if (sYaw)
	{
		for (int32 k = 0; k < sYaw->Size; ++k)
		{
			const int32 idx = sYaw->Start + k;
			OutRaw[idx] = FMath::Clamp(OutRaw[idx], -1.f, 1.f) * PI;
		}
	}

	if (sRV)
	{
		check(TanhScalesRootVel.Num() == sRV->Size);
		for (int32 k = 0; k < sRV->Size; ++k)
		{
			const int32 idx = sRV->Start + k;
			const float Scale = FMath::Max(FMath::Abs(TanhScalesRootVel[k]), 1e-3f);
			OutRaw[idx] = Scale * AtanhSafe(FMath::Clamp(OutRaw[idx], -1.f, 1.f));
		}
	}

	if (sAV)
	{
		check(TanhScalesAngVel.Num() == sAV->Size);
		for (int32 k = 0; k < sAV->Size; ++k)
		{
			const int32 idx = sAV->Start + k;
			const float Scale = FMath::Max(FMath::Abs(TanhScalesAngVel[k]), 1e-3f);
			OutRaw[idx] = Scale * AtanhSafe(FMath::Clamp(OutRaw[idx], -1.f, 1.f));
		}
	}
}


void UEnemyAnimInstance::DenormY_Z_To_Raw(const TArray<float>& Y_norm, TArray<float>& OutRaw) const
{
    OutRaw.SetNum(OutputDim);

    // 1) μ/σ 反变换
    for (int32 i=0; i<OutputDim; ++i)
    {
        const float mu = MuY.IsValidIndex(i)  ? MuY[i]  : 0.f;
        const float sd = StdY.IsValidIndex(i) ? StdY[i] : 1.f; // 模板已含稳健楼板；此处不再额外 floor
        const float zn = Y_norm.IsValidIndex(i) ? Y_norm[i] : 0.f;
        OutRaw[i] = zn * sd + mu;
    }

    // 2) 若未来 Y 重新包含特殊通道，则做逆变换（当前 v4 通常不会命中）
    const FStateSlice* sYaw = OutputLayout.Find(TEXT("RootYaw"));
    const FStateSlice* sRV  = OutputLayout.Find(TEXT("RootVelocity"));
    const FStateSlice* sAV  = OutputLayout.Find(TEXT("BoneAngularVelocities"));

    // RootYaw: [-1,1] → 弧度
    if (sYaw)
    {
        for (int32 k=0; k<sYaw->Size; ++k)
        {
            const int32 idx = sYaw->Start + k;
            OutRaw[idx] = FMath::Clamp(OutRaw[idx], -1.f, 1.f) * PI;
        }
    }

    // RootVelocity: tanh 的逆（atanh * scale）
    if (sRV)
    {
        check(TanhScalesRootVel.Num() == sRV->Size);
        for (int32 k=0; k<sRV->Size; ++k)
        {
            const int32 idx = sRV->Start + k;
            OutRaw[idx]     = TanhScalesRootVel[k] * AtanhSafe(FMath::Clamp(OutRaw[idx], -1.f, 1.f));
        }
    }

    // BoneAngularVelocities: tanh 的逆
    if (sAV)
    {
        check(TanhScalesAngVel.Num() == sAV->Size);
        for (int32 k=0; k<sAV->Size; ++k)
        {
            const int32 idx = sAV->Start + k;
            OutRaw[idx]     = TanhScalesAngVel[k] * AtanhSafe(FMath::Clamp(OutRaw[idx], -1.f, 1.f));
        }
	}
}

void UEnemyAnimInstance::BuildPoseFromRawState(const TArray<float>& RawState, TArray<FTransform>& OutPoseSrc) const
{
	const FStateSlice* RotSlice = StateLayout.Find(TEXT("BoneRotations6D"));
	if (!RotSlice || RotSlice->Size <= 0)
	{
		OutPoseSrc.Reset();
		return;
	}

	const int32 NumBones = RotSlice->Size / 6;
	OutPoseSrc.SetNum(NumBones);
	const bool bUseXZ = (DecodeSpec.Mode == ESixDMode::XZ);

	for (int32 Bone = 0; Bone < NumBones; ++Bone)
	{
		const int32 Base = RotSlice->Start + Bone * 6;
		const float C0X = RawState.IsValidIndex(Base + 0) ? RawState[Base + 0] : 1.f;
		const float C0Y = RawState.IsValidIndex(Base + 1) ? RawState[Base + 1] : 0.f;
		const float C0Z = RawState.IsValidIndex(Base + 2) ? RawState[Base + 2] : 0.f;
		const float C1X = RawState.IsValidIndex(Base + 3) ? RawState[Base + 3] : 0.f;
		const float C1Y = RawState.IsValidIndex(Base + 4) ? RawState[Base + 4] : 1.f;
		const float C1Z = RawState.IsValidIndex(Base + 5) ? RawState[Base + 5] : 0.f;

		const FVector C0(C0X, C0Y, C0Z);
		const FVector C1(C1X, C1Y, C1Z);
		const FQuat Q = SixDToQuat_BySchema(C0, C1, bUseXZ);
		OutPoseSrc[Bone] = FTransform(Q, FVector::ZeroVector, FVector::OneVector);
	}
}

float UEnemyAnimInstance::AtanhSafe(float x) const
{
	const float eps = 1e-6f;
	const float xc  = FMath::Clamp(x, -1.f + eps, 1.f - eps);
	return 0.5f * FMath::Loge((1.f + xc) / (1.f - xc));
}

void UEnemyAnimInstance::ResetCondHistory()
{
	CondHistoryCursor = 0;
	CondHistoryCount = 0;
	const int32 Dim = FMath::Max(CondDim, 0);
	const int32 Window = FMath::Max(CondNormWindow, 0);
	if (Dim > 0 && Window > 0)
	{
		CondHistory.SetNumZeroed(Dim * Window);
	}
	else
	{
		CondHistory.Reset();
	}
}

void UEnemyAnimInstance::PushCondHistorySample(const TArray<float>& Sample)
{
	const int32 Dim = FMath::Max(CondDim, 0);
	const int32 Window = FMath::Max(CondNormWindow, 0);
	if (Dim <= 0 || Window <= 0)
	{
		return;
	}
	const int32 ExpectedSize = Dim * Window;
	if (CondHistory.Num() != ExpectedSize)
	{
		CondHistory.SetNumZeroed(ExpectedSize);
		CondHistoryCursor = 0;
		CondHistoryCount = 0;
	}

	float* Dest = CondHistory.GetData() + CondHistoryCursor * Dim;
	FMemory::Memzero(Dest, sizeof(float) * Dim);
	for (int32 i = 0; i < Dim && i < Sample.Num(); ++i)
	{
		Dest[i] = Sample[i];
	}
	CondHistoryCursor = (CondHistoryCursor + 1) % Window;
	CondHistoryCount = FMath::Min(CondHistoryCount + 1, Window);
}

bool UEnemyAnimInstance::ComputeCondWindowStats(TArray<float>& OutMu, TArray<float>& OutStd) const
{
	const int32 Dim = FMath::Max(CondDim, 0);
	if (Dim <= 0 || CondHistoryCount <= 0 || CondHistory.Num() != Dim * FMath::Max(CondNormWindow, 0))
	{
		return false;
	}

	OutMu.Init(0.f, Dim);
	OutStd.Init(1.f, Dim);

	TArray<float> Samples;
	Samples.SetNum(CondHistoryCount);
	for (int32 d = 0; d < Dim; ++d)
	{
		for (int32 t = 0; t < CondHistoryCount; ++t)
		{
			const int32 Slot = (CondHistoryCursor - CondHistoryCount + t + FMath::Max(CondNormWindow, 0)) % FMath::Max(CondNormWindow, 1);
			const int32 Index = Slot * Dim + d;
			Samples[t] = CondHistory.IsValidIndex(Index) ? CondHistory[Index] : 0.f;
		}

		TArray<float> SortedSamples = Samples;
		const float Q1 = ComputePercentile(SortedSamples, 25.f);
		const float Q3 = ComputePercentile(SortedSamples, 75.f);
		const float IQR = Q3 - Q1;
		const float Lo = Q1 - 1.5f * IQR;
		const float Hi = Q3 + 1.5f * IQR;

		double Sum = 0.0;
		double SumSq = 0.0;
		int32 Count = 0;
		for (int32 t = 0; t < CondHistoryCount; ++t)
		{
			const float V = Samples[t];
			if (V >= Lo && V <= Hi)
			{
				Sum += V;
				SumSq += double(V) * double(V);
				++Count;
			}
		}
		if (Count == 0)
		{
			for (int32 t = 0; t < CondHistoryCount; ++t)
			{
				const float V = Samples[t];
				Sum += V;
				SumSq += double(V) * double(V);
			}
			Count = CondHistoryCount;
		}
		const double Mean = (Count > 0) ? (Sum / Count) : 0.0;
		const double Var = (Count > 0) ? FMath::Max((SumSq / Count) - Mean * Mean, 1e-6) : 1e-6;
		OutMu[d] = (float)Mean;
		OutStd[d] = FMath::Max((float)FMath::Sqrt(Var), 1e-3f);
	}
	return true;
}

void UEnemyAnimInstance::ApplyCondNormalization(const TArray<float>& RawCond, TArray<float>& OutCond)
{
	const int32 Dim = FMath::Max(CondDim, 0);
	if (Dim <= 0)
	{
		OutCond.Reset();
		return;
	}

	auto CopyRaw = [&]()
	{
		OutCond.SetNum(Dim);
		for (int32 i = 0; i < Dim; ++i)
		{
			OutCond[i] = RawCond.IsValidIndex(i) ? RawCond[i] : 0.f;
		}
	};

	if (CondNormMode == ECondNormMode::ZScore)
	{
		TArray<float> Mu, Std;
		if (ComputeCondWindowStats(Mu, Std))
		{
			OutCond.SetNum(Dim);
			for (int32 i = 0; i < Dim; ++i)
			{
				const float val = RawCond.IsValidIndex(i) ? RawCond[i] : 0.f;
				const float mu = Mu.IsValidIndex(i) ? Mu[i] : 0.f;
				const float sd = Std.IsValidIndex(i) ? FMath::Max(Std[i], 1e-3f) : 1.f;
				OutCond[i] = (val - mu) / sd;
			}
			return;
		}
	}

	// RAW 模式或统计不足：直接返回原值
	CopyRaw();
}

bool UEnemyAnimInstance::IsTeacherModelDriving() const
{
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
	return bEnableTeacherPlayback && bTeacherClipReady && bUseTeacherForcingEval && NeuralNetwork.IsValid() && NeuralNetwork->IsReady();
#else
	return false;
#endif
}

void UEnemyAnimInstance::PublishTeacherFallbackPose()
{
	if (LastTeacherPoseSrc.Num() == kTrackedBones.Num() && Prev_X_raw.Num() == InputDim)
	{
		PoseBox.Publish(LastTeacherPoseSrc, Prev_X_raw, /*bLog=*/false, GFrameNumber);
	}
}



FVector2D UEnemyAnimInstance::ResolvePlanarDirection(float& OutSpeedMs) const
{
	OutSpeedMs = 0.f;
	const APawn* Pawn = TryGetPawnOwner();
	if (!Pawn)
	{
		return FVector2D(0.f, 1.f);
	}

	const FVector Velocity = Pawn->GetVelocity();
	const FVector2D Planar(Velocity.X, Velocity.Y);
	OutSpeedMs = Planar.Size() * 0.01f;

	FVector2D Dir = Planar.GetSafeNormal();
	if (Dir.IsNearlyZero())
	{
		const FVector Forward = Pawn->GetActorForwardVector();
		Dir = FVector2D(Forward.X, Forward.Y).GetSafeNormal();
		if (Dir.IsNearlyZero())
		{
			Dir = FVector2D(0.f, 1.f);
		}
	}

	if (bEnableWanderOffset && !Dir.IsNearlyZero())
	{
		const double TimeSec = GetWorld() ? GetWorld()->GetTimeSeconds() : 0.0;
		const float Phase = (float)(TimeSec * WanderSpeed * 2.0 * PI);
		const float Angle = MaxWanderAngle * FMath::Sin(Phase);
		Dir = Dir.GetRotated(Angle);
		Dir.Normalize();
	}

	return Dir;
}


#if ENEMY_ANIM_WITH_ONNX_RUNTIME
void UEnemyAnimInstance::LoadOnnxModel()
{
	NeuralNetwork.Reset();

	if (!OnnxModelDataAsset)
	{
		UE_LOG(LogTemp, Error, TEXT("OnnxModelDataAsset is not specified in the Animation Blueprint."));
		return;
	}

	const FFusedModelInputDims InputDims = BuildModelInputDims();

	NeuralNetwork = TUniquePtr<FFusedEventMotionModel, FFusedEventMotionModelDeleter>(new FFusedEventMotionModel());
	if (!NeuralNetwork->Initialize(OnnxModelDataAsset, InputDims))
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to initialize EventMotionModel ONNX runtime."));
		NeuralNetwork.Reset();
		return;
	}

	const FFusedModelInputDims& BoundDims = NeuralNetwork->GetBoundInputDims();
	EnsurePoseHistoryConfig(BoundDims.PoseHist);

	UE_LOG(LogTemp, Log, TEXT("EventMotionModel loaded (stateless MLP inference)."));
}
#endif

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
void UEnemyAnimInstance::EnsurePoseHistoryConfig(int32 RequiredPoseHistDim)
{
	if (RequiredPoseHistDim <= 0)
	{
		return;
	}
	if (PoseHistoryLen > 0 && PoseHistoryScales.Num() > 0)
	{
		return;
	}

	const int32 Bones = kTrackedBones.Num();
	if (Bones <= 0)
	{
		return;
	}

	const int32 Denominator = FMath::Max(1, Bones * 6);
	const int32 Frames = FMath::Max(1, RequiredPoseHistDim / Denominator);
	PoseHistoryLen = Frames;
	const int32 TargetDim = PoseHistoryLen * Bones * 6;

	auto ResizeAndFill = [&](TArray<float>& Arr, float Value)
	{
		Arr.SetNum(TargetDim);
		for (int32 i = 0; i < TargetDim; ++i)
		{
			Arr[i] = Value;
		}
	};

	ResizeAndFill(PoseHistoryScales, 1.f);
	ResizeAndFill(PoseHistoryMu, 0.f);
	ResizeAndFill(PoseHistoryStd, 1.f);

	InitializePoseHistory(PoseHistoryLen);

	TArray<FTransform> IdentityPose;
	IdentityPose.Init(FTransform::Identity, Bones);
	ResetPoseHistoryFromPose(IdentityPose);

	UE_LOG(LogTemp, Warning, TEXT("[PoseHist] Fallback enabled: Frames=%d Dim=%d (runtime required %d)."),
		PoseHistoryLen, TargetDim, RequiredPoseHistDim);
}
#endif

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
FFusedModelInputDims UEnemyAnimInstance::BuildModelInputDims() const
{
	FFusedModelInputDims Dims;
	Dims.Motion = FMath::Max(InputDim, 1);
	Dims.Cond = FMath::Max(CondDim, 0);

	if (const FStateSlice* ContactsSlice = StateLayout.Find(TEXT("Contacts")))
	{
		Dims.Contacts = FMath::Max(ContactsSlice->Size, 0);
	}
	if (const FStateSlice* AngSlice = StateLayout.Find(TEXT("BoneAngularVelocities")))
	{
		Dims.AngVel = FMath::Max(AngSlice->Size, 0);
	}

	const int32 Bones = kTrackedBones.Num();
	const int32 PoseHistDimFromScales = PoseHistoryScales.Num() > 0 ? PoseHistoryScales.Num() : 0;
	const int32 PoseHistDimFallback = (PoseHistoryLen > 0 && Bones > 0) ? (PoseHistoryLen * Bones * 6) : 0;
	Dims.PoseHist = FMath::Max(PoseHistDimFromScales > 0 ? PoseHistDimFromScales : PoseHistDimFallback, 0);

	// Contacts/AngVel/PoseHist can legally be zero if未启用，对应输入会绑定空张量。
	return Dims;
}
#endif

/**
 * @brief 执行单步模型推理 (每帧或每个固定时间步调用)
 */
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
bool UEnemyAnimInstance::StepModel(float DeltaSeconds)
{
	if (!NeuralNetwork.IsValid() || !NeuralNetwork->IsReady())
	{
		return false;
	}
	return StepModelFused(DeltaSeconds);
}

bool UEnemyAnimInstance::StepModelFused(float DeltaSeconds)
{
	using namespace UE::NNE;

	if (!NeuralNetwork.IsValid() || !NeuralNetwork->IsReady())
	{
		return false;
	}
	if (InputDim <= 0 || OutputDim <= 0)
	{
		return false;
	}
	if (CurrentMotionStateNorm.Num() != InputDim)
	{
		return false;
	}

	if (PoseHistoryLen > 0 && PoseHistoryBuffer.Num() == 0)
	{
		TArray<FTransform> IdentityPose;
		IdentityPose.Init(FTransform::Identity, kTrackedBones.Num());
		ResetPoseHistoryFromPose(IdentityPose);
	}

	TArray<float> CondVec;
	CondVec.Init(0.f, CondDim);
	BuildCondVector(CondVec);

	TArray<float> ContactsVec;
	ExtractStateSlice(StateLayout.Find(TEXT("Contacts")), CurrentMotionStateNorm, ContactsVec);

	TArray<float> AngVelVec;
	ExtractStateSlice(StateLayout.Find(TEXT("BoneAngularVelocities")), CurrentMotionStateNorm, AngVelVec);

	TArray<float> PoseHistVec;
	BuildPoseHistoryFeature(PoseHistVec);

	if (CurrentTeacherContactsNorm.Num() > 0)
	{
		ContactsVec = CurrentTeacherContactsNorm;
	}
	if (CurrentTeacherAngVelNorm.Num() > 0)
	{
		AngVelVec = CurrentTeacherAngVelNorm;
	}
	if (CurrentTeacherPoseHistNorm.Num() > 0)
	{
		PoseHistVec = CurrentTeacherPoseHistNorm;
	}

	auto AlignInputVec = [](TArray<float>& Vec, int32 Dim)
	{
		if (Dim <= 0)
		{
			Vec.Reset();
			return;
		}
		const int32 OldNum = Vec.Num();
		if (OldNum == Dim)
		{
			return;
		}
		Vec.SetNum(Dim);
		if (Dim > OldNum)
		{
			for (int32 i = OldNum; i < Dim; ++i)
			{
				Vec[i] = 0.f;
			}
		}
	};

	const FFusedModelInputDims BoundDims = NeuralNetwork->GetBoundInputDims();
	AlignInputVec(CurrentMotionStateNorm, BoundDims.Motion);
	AlignInputVec(CondVec, BoundDims.Cond);
	AlignInputVec(ContactsVec, BoundDims.Contacts);
	AlignInputVec(AngVelVec, BoundDims.AngVel);
	AlignInputVec(PoseHistVec, BoundDims.PoseHist);

	const bool bTeacherDriving = IsTeacherModelDriving();

	static int32 DebugStepModelLogs = 0;
	if (DebugStepModelLogs < 8)
	{
		UE_LOG(LogTemp, Warning, TEXT("[TDbg][StepModel] lens X=%d C=%d Contacts=%d Ang=%d Pose=%d | bound %d/%d/%d/%d/%d | teacher=%d"),
			CurrentMotionStateNorm.Num(), CondVec.Num(), ContactsVec.Num(), AngVelVec.Num(), PoseHistVec.Num(),
			BoundDims.Motion, BoundDims.Cond, BoundDims.Contacts, BoundDims.AngVel, BoundDims.PoseHist, (int32)bTeacherDriving);
		++DebugStepModelLogs;
	}

	TArray<float> MotionNorm;
	if (!NeuralNetwork->Forward(CurrentMotionStateNorm, CondVec, ContactsVec, AngVelVec, PoseHistVec, MotionNorm))
	{
		return false;
	}

	if (MotionNorm.Num() != OutputDim)
	{
		if (MotionNorm.Num() == 0)
		{
			return false;
		}
		UE_LOG(LogTemp, Warning, TEXT("[FusedModel] Output dimension mismatch: expected %d, got %d"), OutputDim, MotionNorm.Num());
	}

	TArray<float> MotionDenorm;
	DenormY_Z_To_Raw(MotionNorm, MotionDenorm);

	const FStateSlice* SY_rot = OutputLayout.Find(TEXT("BoneRotations6D"));
	if (!SY_rot)
	{
		UE_LOG(LogTemp, Error, TEXT("[FusedModel] output_layout lacks BoneRotations6D."));
		return false;
	}

	auto Reproject6D = [&](const TArray<float>& InOut, TArray<float>& OutRot) -> void
	{
		OutRot = InOut;
		const int32 NBlocks = SY_rot->Size / 6;
		if (AngVelPrevQ.Num() != NBlocks)
		{
			AngVelPrevQ.Init(FQuat::Identity, NBlocks);
		}
		auto Reproject6D_SO3_XZ = [&](const FVector& C0, const FVector& C1, const FQuat& PrevQ)->TPair<FVector, FVector>
		{
			FVector X = C0.GetSafeNormal(); if (X.IsNearlyZero()) X = PrevQ.GetAxisX();
			FVector Z = (C1 - FVector::DotProduct(C1, X)*X).GetSafeNormal(); if (Z.IsNearlyZero()) Z = PrevQ.GetAxisZ();
			FVector Y = FVector::CrossProduct(Z, X).GetSafeNormal();
			if (FVector::DotProduct(X, FVector::CrossProduct(Y, Z)) < 0.f) Y *= -1.f;
			Z = FVector::CrossProduct(X, Y).GetSafeNormal();
			return {X, Z};
		};

		for (int32 b = 0; b < NBlocks; ++b)
		{
			const int yi = SY_rot->Start + b * 6;
			const FVector C0(OutRot[yi + 0], OutRot[yi + 1], OutRot[yi + 2]);
			const FVector C1(OutRot[yi + 3], OutRot[yi + 4], OutRot[yi + 5]);
			const FQuat   Qp = AngVelPrevQ.IsValidIndex(b) ? AngVelPrevQ[b] : FQuat::Identity;
			const auto XZ = Reproject6D_SO3_XZ(C0, C1, Qp);
			OutRot[yi + 0] = XZ.Key.X;   OutRot[yi + 1] = XZ.Key.Y;   OutRot[yi + 2] = XZ.Key.Z;
			OutRot[yi + 3] = XZ.Value.X; OutRot[yi + 4] = XZ.Value.Y; OutRot[yi + 5] = XZ.Value.Z;
		}
	};

	auto ComposeDeltaWithPrev = [&]() -> void
	{
		const FStateSlice* SX_rot = StateLayout.Find(TEXT("BoneRotations6D"));
		if (!SX_rot || SX_rot->Size != SY_rot->Size || SX_rot->Size <= 0)
		{
			return;
		}
		if (Prev_X_raw.Num() < SX_rot->Start + SX_rot->Size)
		{
			return;
		}
		const int32 NBlocks = SY_rot->Size / 6;
		for (int32 b = 0; b < NBlocks; ++b)
		{
			const int32 PrevIdx = SX_rot->Start + b * 6;
			const int32 DeltaIdx = SY_rot->Start + b * 6;
			float DeltaRaw[6];
			for (int32 k = 0; k < 6; ++k)
			{
				DeltaRaw[k] = MotionDenorm.IsValidIndex(DeltaIdx + k) ? MotionDenorm[DeltaIdx + k] : 0.f;
			}
			FMatrix PrevM, DeltaM;
			DecodeRot6DToMatrix(&Prev_X_raw[PrevIdx], PrevM);
			DecodeRot6DToMatrix(DeltaRaw, DeltaM);
			const FMatrix NextM = DeltaM * PrevM;
			EncodeMatrixToRot6D(NextM, &MotionDenorm[DeltaIdx]);

			if (b == 0)
			{
				static int32 DebugComposeLogs = 0;
				if (DebugComposeLogs < 4)
				{
					const FVector NewX = NextM.GetScaledAxis(EAxis::X).GetSafeNormal();
					const FVector NewZ = NextM.GetScaledAxis(EAxis::Z).GetSafeNormal();
					UE_LOG(LogTemp, Warning, TEXT("[DeltaCompose] frame=%u prevX=(%.3f,%.3f,%.3f) prevZ=(%.3f,%.3f,%.3f) deltaX=(%.3f,%.3f,%.3f) deltaZ=(%.3f,%.3f,%.3f) nextX=(%.3f,%.3f,%.3f) nextZ=(%.3f,%.3f,%.3f)"),
						GFrameNumber,
						Prev_X_raw[PrevIdx + 0], Prev_X_raw[PrevIdx + 1], Prev_X_raw[PrevIdx + 2],
						Prev_X_raw[PrevIdx + 3], Prev_X_raw[PrevIdx + 4], Prev_X_raw[PrevIdx + 5],
						DeltaRaw[0], DeltaRaw[1], DeltaRaw[2],
						DeltaRaw[3], DeltaRaw[4], DeltaRaw[5],
						NewX.X, NewX.Y, NewX.Z, NewZ.X, NewZ.Y, NewZ.Z);
					++DebugComposeLogs;
				}
			}
		}
	};

	ComposeDeltaWithPrev();
	TArray<float> MotionReprojected;
	Reproject6D(MotionDenorm, MotionReprojected);

	const int32 NBlocks = SY_rot->Size / 6;
	PredictedLocal_Src.SetNum(NBlocks);
	AngVelPrevQ.SetNum(NBlocks);
	for (int32 b = 0; b < NBlocks; ++b)
	{
		const int yi = SY_rot->Start + b * 6;
		FVector Xv(MotionReprojected[yi + 0], MotionReprojected[yi + 1], MotionReprojected[yi + 2]);
		FVector Zv(MotionReprojected[yi + 3], MotionReprojected[yi + 4], MotionReprojected[yi + 5]);
		Xv = Xv.GetSafeNormal(); if (Xv.IsNearlyZero()) Xv = FVector::ForwardVector;
		Zv = (Zv - FVector::DotProduct(Zv, Xv)*Xv).GetSafeNormal(); if (Zv.IsNearlyZero()) Zv = FVector::UpVector;
		FVector Yv = FVector::CrossProduct(Zv, Xv).GetSafeNormal();
		if (FVector::DotProduct(Xv, FVector::CrossProduct(Yv, Zv)) < 0.f) Yv *= -1.f;
		Zv = FVector::CrossProduct(Xv, Yv).GetSafeNormal();
		FMatrix M = FMatrix::Identity; M.SetAxis(0, Xv); M.SetAxis(1, Yv); M.SetAxis(2, Zv);
		FQuat Q(M); Q.Normalize();
		PredictedLocal_Src[b].SetRotation(Q);
		PredictedLocal_Src[b].SetTranslation(FVector::ZeroVector);
		AngVelPrevQ[b] = Q;
	}

	// 源坐标 -> UE 坐标（与 Seed/Teacher 路径一致）
	if (PredictedLocal_UE.Num() != NBlocks)
	{
		PredictedLocal_UE.SetNum(NBlocks);
	}
	for (int32 b = 0; b < NBlocks; ++b)
	{
		FQuat Qsrc = PredictedLocal_Src[b].GetRotation();
		if (!Qsrc.IsNormalized())
		{
			Qsrc.Normalize();
		}
		FQuat Que = bConjugateS2U ? SafeQ(CachedQ_SrcToUE * Qsrc * CachedQ_UEToSrc) : Qsrc;
		PredictedLocal_UE[b].SetRotation(Que);
		PredictedLocal_UE[b].SetTranslation(PredictedLocal_Src[b].GetTranslation());
	}

	PoseBox.Publish(PredictedLocal_UE, MotionReprojected, /*bRootIncluded*/false, GFrameNumber);

	// Debug: print first few elements of y prediction (denorm, reprojected)
	static int32 DebugYLogs = 0;
	if (DebugYLogs < 8)
	{
		FString Slice;
		for (int32 i = 0; i < MotionReprojected.Num() && i < 12; ++i)
		{
			Slice += FString::Printf(TEXT("%s%.4f"), (i == 0 ? TEXT("") : TEXT(",")), MotionReprojected[i]);
		}
		UE_LOG(LogTemp, Warning, TEXT("[TDbg][StepModel] y_reproj[0..11]=%s (len=%d)"), *Slice, MotionReprojected.Num());
		++DebugYLogs;
	}

	if (!bTeacherDriving)
	{
		PushPoseHistoryFrame(PredictedLocal_Src);

		if (Prev_X_raw.Num() != InputDim)
		{
			Prev_X_raw = MotionReprojected;
			Prev_X_raw.SetNum(InputDim);
		}
		else
		{
			for (const TPair<FString, FStateSlice>& Pair : OutputLayout)
			{
				const FStateSlice* SX = StateLayout.Find(Pair.Key);
				if (!SX) continue;
				const FStateSlice& SliceOut = Pair.Value;
				if (SX->Size != SliceOut.Size) continue;

				for (int32 k = 0; k < SliceOut.Size; ++k)
				{
					const int32 SrcIdx = SliceOut.Start + k;
					const int32 DstIdx = SX->Start + k;
					if (MotionReprojected.IsValidIndex(SrcIdx) && Prev_X_raw.IsValidIndex(DstIdx))
					{
						Prev_X_raw[DstIdx] = MotionReprojected[SrcIdx];
					}
				}
			}
		}

		NormalizeXRaw_To_Z(Prev_X_raw, CurrentMotionStateNorm);
		CarryX_Norm = CurrentMotionStateNorm;
	}

	return true;
}
#endif

// 将函数名稍作修改以表明其职责，并让它接收一个数组的引用
/**
 * @brief (重构后) 辅助函数：根据骨骼参考姿态，计算6D旋转值并写入传入的RawState数组。
 * @param InOutRawState 一个引用，函数将直接修改这个数组中 BoneRotations6D 对应的部分。
 */
void UEnemyAnimInstance::WarmStartPoseInRawState(TArray<float>& InOutRawState)
{
    // --- 1. 获取必要的组件和数据 ---
    USkeletalMeshComponent* SKC = GetSkelMeshComponent();
    if (!SKC || !SKC->GetSkeletalMeshAsset())
    {
        UE_LOG(LogTemp, Warning, TEXT("WarmStartPose skipped: invalid skeletal mesh."));
        return;
    }

    const USkeletalMesh* Skel = SKC->GetSkeletalMeshAsset();
    const FReferenceSkeleton& RefSkel = Skel->GetRefSkeleton();
    const TArray<FTransform>& RefPose = RefSkel.GetRefBonePose();

    // --- 2. 找到 BoneRotations6D 在状态向量中的位置 ---
    const FStateSlice* SRot = StateLayout.Find(TEXT("BoneRotations6D"));
    if (!SRot || SRot->Size <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("WarmStartPose skipped: state_layout lacks BoneRotations6D."));
        return;
    }
    const int32 RotStart = SRot->Start;
    const int32 RotSize  = SRot->Size;
    const int32 NumBonesByLayout = RotSize / 6;

    // --- 3. 确定骨骼顺序和6D编码规格 ---
    const int32 NumTracked = kTrackedBones.Num() > 0 ? kTrackedBones.Num() : NumBonesByLayout;
    const int32 NumToWrite = FMath::Min(NumTracked, NumBonesByLayout);
    
    // 从您的解码规格中获取6D编码模式 (XY, XZ, etc.)
    const ESixDMode Mode = DecodeSpec.Mode;
    const bool bUseXZ = (Mode == ESixDMode::XZ);

    // --- 4. 遍历骨骼，计算6D旋转并写入传入的数组 ---
    for (int32 i = 0; i < NumToWrite; ++i)
    {
        // 确定当前模型骨骼索引对应的参考骨架中的索引
        const int32 RefIdx = (kTrackedBones.Num() > 0 && kTrackedBones.IsValidIndex(i))
                                 ? RefSkel.FindBoneIndex(kTrackedBones[i])
                                 : i;

        if (!RefPose.IsValidIndex(RefIdx))
        {
            // 如果BoneNamesInModelOrder中的骨骼名在参考骨架中找不到，RefIdx会是-1
            if (kTrackedBones.IsValidIndex(i))
            {
                UE_LOG(LogTemp, Warning, TEXT("[WarmStart] Bone '%s' not found in RefSkeleton. Skipping."), *kTrackedBones[i].ToString());
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("[WarmStart] Invalid RefPose index at i=%d"), i);
            }
            continue;
        }

        // 获取UE的局部空间旋转
        const FQuat Q_UE_Local = SafeQ(RefPose[RefIdx].GetRotation());
        
        // 如果需要，将UE旋转转换回源坐标系再进行6D编码
        const FQuat Q_Src_Local = bConjugateS2U ? SafeQ(CachedQ_UEToSrc * Q_UE_Local * CachedQ_SrcToUE)
                                                : Q_UE_Local;

        // 将源空间的旋转四元数编码为6D向量
        FVector C0(1,0,0), C1(0,1,0); // 默认值
        QuatTo6D_BySchema(Q_Src_Local, bUseXZ, C0, C1);

        // 直接写入传入的 InOutRawState 数组的正确位置
        const int32 r0 = RotStart + i * 6;
        if (InOutRawState.IsValidIndex(r0 + 5))
        {
            InOutRawState[r0 + 0] = C0.X; InOutRawState[r0 + 1] = C0.Y; InOutRawState[r0 + 2] = C0.Z;
            InOutRawState[r0 + 3] = C1.X; InOutRawState[r0 + 4] = C1.Y; InOutRawState[r0 + 5] = C1.Z;
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("[WarmStartPose] Pose portion of raw state initialized from RefPose."));
}

void UEnemyAnimInstance::InitializePoseHistory(int32 InHistoryLen)
{
	const int32 Bones = kTrackedBones.Num();
	PoseHistoryLen = FMath::Max(0, InHistoryLen);
	PoseHistoryHead = 0;
	PoseHistoryValidFrames = 0;

	if (PoseHistoryLen <= 0 || Bones <= 0)
	{
		PoseHistoryBuffer.Reset();
		return;
	}

	PoseHistoryBuffer.SetNum(PoseHistoryLen * Bones);
	for (FTransform& T : PoseHistoryBuffer)
	{
		T = FTransform::Identity;
	}
}

void UEnemyAnimInstance::ResetPoseHistoryFromPose(const TArray<FTransform>& PoseSrcSpace)
{
	if (PoseHistoryLen <= 0)
	{
		return;
	}

	const int32 Bones = kTrackedBones.Num();
	if (PoseHistoryBuffer.Num() != PoseHistoryLen * Bones)
	{
		InitializePoseHistory(PoseHistoryLen);
	}

	PoseHistoryHead = 0;
	PoseHistoryValidFrames = 0;
	for (int32 i = 0; i < PoseHistoryLen; ++i)
	{
		PushPoseHistoryFrame(PoseSrcSpace);
	}
}

void UEnemyAnimInstance::PushPoseHistoryFrame(const TArray<FTransform>& PoseSrcSpace)
{
	const int32 Bones = kTrackedBones.Num();
	if (PoseHistoryLen <= 0 || Bones <= 0)
	{
		return;
	}
	if (PoseHistoryBuffer.Num() != PoseHistoryLen * Bones)
	{
		InitializePoseHistory(PoseHistoryLen);
	}

	const int32 FrameIndex = PoseHistoryHead;
	for (int32 BoneIdx = 0; BoneIdx < Bones; ++BoneIdx)
	{
		const int32 BufferIdx = FrameIndex * Bones + BoneIdx;
		if (!PoseHistoryBuffer.IsValidIndex(BufferIdx))
		{
			continue;
		}

		if (PoseSrcSpace.IsValidIndex(BoneIdx))
		{
			PoseHistoryBuffer[BufferIdx] = PoseSrcSpace[BoneIdx];
		}
		else
		{
			PoseHistoryBuffer[BufferIdx] = FTransform::Identity;
		}
	}

	PoseHistoryHead = (PoseHistoryHead + 1) % PoseHistoryLen;
	PoseHistoryValidFrames = FMath::Min(PoseHistoryValidFrames + 1, PoseHistoryLen);
}

void UEnemyAnimInstance::BuildPoseHistoryFeature(TArray<float>& OutFeature) const
{
	const int32 Bones = kTrackedBones.Num();
	if (PoseHistoryLen <= 0 || Bones <= 0)
	{
		OutFeature.Reset();
		return;
	}

	const int32 StridePerFrame = Bones * 6;
	const int32 ExpectedDim = PoseHistoryLen * StridePerFrame;
	const int32 FeatureDim = PoseHistoryScales.Num() > 0 ? PoseHistoryScales.Num() : ExpectedDim;

	if (FeatureDim <= 0)
	{
		OutFeature.Reset();
		return;
	}

	TArray<float> Raw;
	Raw.SetNumZeroed(FeatureDim);

	if (PoseHistoryBuffer.Num() != PoseHistoryLen * Bones)
	{
		NormalizePoseHistoryRaw(Raw, OutFeature);
		return;
	}

	auto ResolveFrameIndex = [&](int32 Slot)->int32
	{
		if (PoseHistoryValidFrames <= 0)
		{
			return PoseHistoryHead;
		}
		if (PoseHistoryValidFrames < PoseHistoryLen)
		{
			const int32 Missing = PoseHistoryLen - PoseHistoryValidFrames;
			if (Slot < Missing)
			{
				return PoseHistoryHead;
			}
			const int32 Offset = Slot - Missing;
			return (PoseHistoryHead + Offset) % PoseHistoryLen;
		}
		return (PoseHistoryHead + Slot) % PoseHistoryLen;
	};

	const bool bUseXZ = (DecodeSpec.Mode == ESixDMode::XZ);
	for (int32 Slot = 0; Slot < PoseHistoryLen; ++Slot)
	{
		const int32 RawBase = Slot * StridePerFrame;
		if (RawBase + StridePerFrame > Raw.Num())
		{
			break;
		}

		const int32 FrameIdx = ResolveFrameIndex(Slot);
		const int32 BufferBase = FrameIdx * Bones;
		for (int32 BoneIdx = 0; BoneIdx < Bones; ++BoneIdx)
		{
			const int32 OutIdx = RawBase + BoneIdx * 6;
			const int32 SrcIdx = BufferBase + BoneIdx;
			const FTransform& Stored = PoseHistoryBuffer.IsValidIndex(SrcIdx) ? PoseHistoryBuffer[SrcIdx] : FTransform::Identity;

			FVector C0, C1;
			QuatTo6D_BySchema(Stored.GetRotation(), bUseXZ, C0, C1);
			Raw[OutIdx + 0] = C0.X;
			Raw[OutIdx + 1] = C0.Y;
			Raw[OutIdx + 2] = C0.Z;
			Raw[OutIdx + 3] = C1.X;
			Raw[OutIdx + 4] = C1.Y;
			Raw[OutIdx + 5] = C1.Z;
		}
	}

	NormalizePoseHistoryRaw(Raw, OutFeature);
}

void UEnemyAnimInstance::NormalizePoseHistoryRaw(const TArray<float>& Raw, TArray<float>& OutNorm) const
{
	if (PoseHistoryScales.Num() <= 0)
	{
		OutNorm = Raw;
		return;
	}

	const int32 Dim = PoseHistoryScales.Num();
	OutNorm.SetNum(Dim);
	for (int32 i = 0; i < Dim; ++i)
	{
		const float Scale = FMath::Max(PoseHistoryScales[i], 1e-6f);
		const float RawVal = Raw.IsValidIndex(i) ? Raw[i] : 0.f;
		float Value = FMath::Tanh(RawVal / Scale);
		if (PoseHistoryMu.IsValidIndex(i) && PoseHistoryStd.IsValidIndex(i))
		{
			const float Std = FMath::Max(PoseHistoryStd[i], 1e-6f);
			Value = (Value - PoseHistoryMu[i]) / Std;
		}
		OutNorm[i] = Value;
	}
}

bool UEnemyAnimInstance::ExtractStateSlice(const FStateSlice* Slice, const TArray<float>& Source, TArray<float>& Out) const
{
	if (!Slice || Slice->Size <= 0 || Slice->Start < 0)
	{
		Out.Reset();
		return false;
	}
	if (Slice->Start + Slice->Size > Source.Num())
	{
		Out.Reset();
		return false;
	}

	Out.SetNum(Slice->Size);
	for (int32 i = 0; i < Slice->Size; ++i)
	{
		Out[i] = Source[Slice->Start + i];
	}
	return true;
}

FString UEnemyAnimInstance::ResolveTeacherClipPath() const
{
	if (!TeacherClipJsonPath.FilePath.IsEmpty())
	{
		return TeacherClipJsonPath.FilePath;
	}
	return StartupSeedJson.FilePath;
}

bool UEnemyAnimInstance::BuildRawStateFromFrameJson(const TSharedPtr<FJsonObject>& FrameObj, TArray<float>& OutRaw, TArray<FTransform>* OutPoseSrc, TArray<float>* OutCondRaw) const
{
	if (!FrameObj.IsValid() || InputDim <= 0)
	{
		return false;
	}

	OutRaw.Init(0.f, InputDim);

	auto CopyVectorField = [&](const TCHAR* Field, const FStateSlice* Slice)->bool
	{
		if (!Slice || Slice->Size <= 0)
		{
			return true;
		}
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (!FrameObj->TryGetArrayField(Field, Arr) || !Arr)
		{
			return false;
		}
		const int32 CopyCount = FMath::Min(Slice->Size, Arr->Num());
		for (int32 i = 0; i < CopyCount; ++i)
		{
			const float Val = (float)(*Arr)[i]->AsNumber();
			if (OutRaw.IsValidIndex(Slice->Start + i))
			{
				OutRaw[Slice->Start + i] = Val;
			}
		}
		return CopyCount == Slice->Size;
	};

	auto CopyScalarField = [&](const TCHAR* Field, const FStateSlice* Slice)->bool
	{
		if (!Slice || Slice->Size <= 0)
		{
			return true;
		}
		double Value = 0.0;
		if (!FrameObj->TryGetNumberField(Field, Value))
		{
			return false;
		}
		if (OutRaw.IsValidIndex(Slice->Start))
		{
			OutRaw[Slice->Start] = (float)Value;
		}
		return true;
	};

	bool bOk = true;

	const FStateSlice* RootPos = StateLayout.Find(TEXT("RootPosition"));
	const FStateSlice* RootVel = StateLayout.Find(TEXT("RootVelocity"));
	const FStateSlice* RootYaw = StateLayout.Find(TEXT("RootYaw"));
	const FStateSlice* RotSlice = StateLayout.Find(TEXT("BoneRotations6D"));
	const FStateSlice* AngSlice = StateLayout.Find(TEXT("BoneAngularVelocities"));

	if (!CopyVectorField(TEXT("RootPosition"), RootPos))
	{
		UE_LOG(LogTemp, Warning, TEXT("[Teacher] Missing RootPosition array in frame."));
		bOk = false;
	}
	if (!CopyVectorField(TEXT("RootVelocityXY"), RootVel))
	{
		UE_LOG(LogTemp, Warning, TEXT("[Teacher] Missing RootVelocityXY array in frame."));
		bOk = false;
	}
	if (!CopyScalarField(TEXT("RootYaw"), RootYaw))
	{
		UE_LOG(LogTemp, Warning, TEXT("[Teacher] Missing RootYaw value in frame."));
		bOk = false;
	}

	const bool bUseXZ = (DecodeSpec.Mode == ESixDMode::XZ);
	if (RotSlice && RotSlice->Size > 0)
	{
		const TArray<TSharedPtr<FJsonValue>>* RotArray = nullptr;
		if (!FrameObj->TryGetArrayField(TEXT("BoneRotations"), RotArray) || !RotArray)
		{
			UE_LOG(LogTemp, Warning, TEXT("[Teacher] Frame missing BoneRotations array."));
			bOk = false;
		}
		else
		{
			const int32 NumBones = RotSlice->Size / 6;
			const int32 UseBones = FMath::Min(NumBones, RotArray->Num());
			if (OutPoseSrc)
			{
				OutPoseSrc->SetNum(NumBones);
				for (int32 i = 0; i < NumBones; ++i)
				{
					(*OutPoseSrc)[i] = FTransform::Identity;
				}
			}

			for (int32 bone = 0; bone < UseBones; ++bone)
			{
				const TSharedPtr<FJsonValue>& BoneValue = (*RotArray)[bone];
				if (!BoneValue.IsValid())
				{
					continue;
				}
				const TArray<TSharedPtr<FJsonValue>>& BoneArray = BoneValue->AsArray();
				const int32 BaseIdx = RotSlice->Start + bone * 6;
				for (int32 c = 0; c < 6 && c < BoneArray.Num(); ++c)
				{
					if (OutRaw.IsValidIndex(BaseIdx + c))
					{
						OutRaw[BaseIdx + c] = (float)BoneArray[c]->AsNumber();
					}
				}

				if (OutPoseSrc && OutPoseSrc->IsValidIndex(bone))
				{
					const FVector C0(
						OutRaw[BaseIdx + 0],
						OutRaw[BaseIdx + 1],
						OutRaw[BaseIdx + 2]);
					const FVector C1(
						OutRaw[BaseIdx + 3],
						OutRaw[BaseIdx + 4],
						OutRaw[BaseIdx + 5]);
					const FQuat Q = SixDToQuat_BySchema(C0, C1, bUseXZ);
					(*OutPoseSrc)[bone] = FTransform(Q, FVector::ZeroVector, FVector::OneVector);
				}
			}
		}
	}

	if (AngSlice && AngSlice->Size > 0)
	{
		const TArray<TSharedPtr<FJsonValue>>* AngArray = nullptr;
		if (FrameObj->TryGetArrayField(TEXT("BoneAngularVelocities"), AngArray) && AngArray)
		{
			const int32 NumBones = AngSlice->Size / 3;
			const int32 UseBones = FMath::Min(NumBones, AngArray->Num());
			for (int32 bone = 0; bone < UseBones; ++bone)
			{
				const TSharedPtr<FJsonValue>& BoneValue = (*AngArray)[bone];
				if (!BoneValue.IsValid())
				{
					continue;
				}
				const TArray<TSharedPtr<FJsonValue>>& BoneArray = BoneValue->AsArray();
				const int32 BaseIdx = AngSlice->Start + bone * 3;
				for (int32 c = 0; c < 3 && c < BoneArray.Num(); ++c)
				{
					if (OutRaw.IsValidIndex(BaseIdx + c))
					{
						OutRaw[BaseIdx + c] = (float)BoneArray[c]->AsNumber();
					}
				}
			}
		}
		else
		{
			UE_LOG(LogTemp, Warning, TEXT("[Teacher] Frame missing BoneAngularVelocities array."));
			bOk = false;
		}
	}

	if (OutCondRaw)
	{
		ExtractCondFromFrameJson(FrameObj, *OutCondRaw);
	}

return bOk;
}

bool UEnemyAnimInstance::ExtractCondFromFrameJson(const TSharedPtr<FJsonObject>& FrameObj, TArray<float>& OutCond) const
{
	const int32 Dim = FMath::Max(CondDim, 0);
	if (!FrameObj.IsValid() || Dim <= 0)
	{
		OutCond.Reset();
		return false;
	}

	OutCond.Init(0.f, Dim);
	bool bAny = false;

	const TArray<TSharedPtr<FJsonValue>>* ActionBitsArray = nullptr;
	if (FrameObj->TryGetArrayField(TEXT("ActionBits"), ActionBitsArray) && ActionBitsArray)
	{
		const int32 CopyCount = FMath::Min(4, ActionBitsArray->Num());
		for (int32 idx = 0; idx < CopyCount; ++idx)
		{
			OutCond[idx] = (float)(*ActionBitsArray)[idx]->AsNumber();
			bAny = true;
		}
	}

	const TArray<TSharedPtr<FJsonValue>>* TrajDirArr = nullptr;
	if (FrameObj->TryGetArrayField(TEXT("TrajectoryDir"), TrajDirArr) && TrajDirArr && TrajDirArr->Num() > 0 && Dim >= 6)
	{
		const int32 Mid = TrajDirArr->Num() / 2;
		const TSharedPtr<FJsonValue>& Entry = (*TrajDirArr)[FMath::Clamp(Mid, 0, TrajDirArr->Num() - 1)];
		if (Entry.IsValid())
		{
			const TArray<TSharedPtr<FJsonValue>> DirPair = Entry->AsArray();
			if (DirPair.Num() >= 2)
			{
				FVector2D Dir(
					static_cast<float>(DirPair[0]->AsNumber()),
					static_cast<float>(DirPair[1]->AsNumber()));
				if (!Dir.IsNearlyZero())
				{
					Dir.Normalize();
				}
				Dir.X = FMath::Clamp(Dir.X, -1.f, 1.f);
				Dir.Y = FMath::Clamp(Dir.Y, -1.f, 1.f);
				OutCond[4] = Dir.X;
				OutCond[5] = Dir.Y;
				bAny = true;
			}
		}
	}

	const TArray<TSharedPtr<FJsonValue>>* RootVelArr = nullptr;
	if (FrameObj->TryGetArrayField(TEXT("RootVelocityXY"), RootVelArr) && RootVelArr && RootVelArr->Num() >= 2 && Dim >= 7)
	{
		const float Vx = (float)(*RootVelArr)[0]->AsNumber();
		const float Vy = (float)(*RootVelArr)[1]->AsNumber();
		const float Speed = FMath::Sqrt(FMath::Square(Vx) + FMath::Square(Vy));
		OutCond[6] = Speed;
		bAny = true;
	}

	return bAny;
}

bool UEnemyAnimInstance::TryBuildRawStateFromSeedJson(TArray<float>& OutXRaw, TArray<FTransform>* OutLocalUE) const
{
	FString ClipPath = StartupSeedJson.FilePath;
	if (ClipPath.IsEmpty())
	{
		ClipPath = ResolveTeacherClipPath();
	}
	if (ClipPath.IsEmpty())
	{
		return false;
	}

	const FString AbsPath = FPaths::ConvertRelativePathToFull(ClipPath);
	FString JsonContent;
	if (!FFileHelper::LoadFileToString(JsonContent, *AbsPath))
	{
		UE_LOG(LogTemp, Warning, TEXT("[SeedJson] Cannot read %s"), *AbsPath);
		return false;
	}

	TSharedPtr<FJsonObject> Root;
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(JsonContent), Root) || !Root.IsValid())
	{
		UE_LOG(LogTemp, Warning, TEXT("[SeedJson] Parse failed: %s"), *AbsPath);
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Frames = nullptr;
	const bool bHasFrames = Root->TryGetArrayField(TEXT("Frames"), Frames) && Frames && Frames->Num() > 0;

	const TSharedPtr<FJsonObject>* TeacherObjPtr = nullptr;
	const bool bHasTeacher = Root->TryGetObjectField(TEXT("teacher"), TeacherObjPtr) && TeacherObjPtr && (*TeacherObjPtr).IsValid();

	if (!bHasFrames && !bHasTeacher)
	{
		UE_LOG(LogTemp, Warning, TEXT("[SeedJson] %s missing Frames/teacher payload."), *AbsPath);
		return false;
	}

	TArray<FTransform> PoseSrc;

	if (bHasFrames)
	{
		const int32 FrameIndex = FMath::Clamp(StartupSeedFrameIndex, 0, Frames->Num() - 1);
		const TSharedPtr<FJsonObject> FrameObj = (*Frames)[FrameIndex]->AsObject();
		if (!FrameObj.IsValid())
		{
			UE_LOG(LogTemp, Warning, TEXT("[SeedJson] Frame %d invalid in %s"), FrameIndex, *AbsPath);
			return false;
		}

		if (!BuildRawStateFromFrameJson(FrameObj, OutXRaw, OutLocalUE ? &PoseSrc : nullptr, nullptr))
		{
			UE_LOG(LogTemp, Warning, TEXT("[SeedJson] Failed to build raw state from %s frame %d"), *AbsPath, FrameIndex);
			return false;
		}
	}
	else
	{
		const TSharedPtr<FJsonObject> TeacherObj = *TeacherObjPtr;
		const TArray<TSharedPtr<FJsonValue>>* StateArray = nullptr;
		if (!TeacherObj->TryGetArrayField(TEXT("state_norm"), StateArray) || !StateArray || StateArray->Num() == 0)
		{
			UE_LOG(LogTemp, Warning, TEXT("[SeedJson] %s teacher payload missing state_norm."), *AbsPath);
			return false;
		}

		const int32 FrameIndex = FMath::Clamp(StartupSeedFrameIndex, 0, StateArray->Num() - 1);
		const TSharedPtr<FJsonValue>& RowVal = (*StateArray)[FrameIndex];
		if (!RowVal.IsValid())
		{
			UE_LOG(LogTemp, Warning, TEXT("[SeedJson] teacher frame %d invalid in %s"), FrameIndex, *AbsPath);
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>> Row = RowVal->AsArray();
		TArray<float> StateNorm;
		StateNorm.Init(0.f, InputDim);
		const int32 CopyCount = FMath::Min(InputDim, Row.Num());
		for (int32 c = 0; c < CopyCount; ++c)
		{
			StateNorm[c] = (float)Row[c]->AsNumber();
		}

		DenormX_Z_To_Raw(StateNorm, OutXRaw);
		if (OutLocalUE)
		{
			BuildPoseFromRawState(OutXRaw, PoseSrc);
		}
	}

	if (OutLocalUE && PoseSrc.Num() > 0)
	{
		const int32 Bones = PoseSrc.Num();
		OutLocalUE->SetNum(Bones);
		for (int32 i = 0; i < Bones; ++i)
		{
			const FQuat Qsrc = PoseSrc[i].GetRotation();
			FQuat Rue = bConjugateS2U ? SafeQ(CachedQ_SrcToUE * Qsrc * CachedQ_UEToSrc) : Qsrc;
			(*OutLocalUE)[i] = FTransform(Rue, FVector::ZeroVector, FVector::OneVector);
		}
	}

	return true;
}

bool UEnemyAnimInstance::LoadTeacherClip()
{
	TeacherFrames.Reset();
	bTeacherClipReady = false;

	if (!bEnableTeacherPlayback)
	{
		return false;
	}

	const FString ClipPath = ResolveTeacherClipPath();
	if (ClipPath.IsEmpty())
	{
		UE_LOG(LogTemp, Warning, TEXT("[Teacher] Clip path not set."));
		return false;
	}

	const FString AbsPath = FPaths::ConvertRelativePathToFull(ClipPath);
	FString JsonContent;
	if (!FFileHelper::LoadFileToString(JsonContent, *AbsPath))
	{
		UE_LOG(LogTemp, Error, TEXT("[Teacher] Cannot read clip %s"), *AbsPath);
		return false;
	}

	TSharedPtr<FJsonObject> Root;
	if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(JsonContent), Root) || !Root.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("[Teacher] Parse failed for %s"), *AbsPath);
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* Frames = nullptr;
	const bool bHasFrames = Root->TryGetArrayField(TEXT("Frames"), Frames) && Frames && Frames->Num() > 0;

	const TSharedPtr<FJsonObject>* TeacherObjPtr = nullptr;
	const bool bHasTeacherObj = Root->TryGetObjectField(TEXT("teacher"), TeacherObjPtr) && TeacherObjPtr && (*TeacherObjPtr).IsValid();

	if (!bHasFrames && !bHasTeacherObj)
	{
		UE_LOG(LogTemp, Error, TEXT("[Teacher] %s missing Frames/teacher payload."), *AbsPath);
		return false;
	}

	double FileFps = TrajHz;
	if (Root->HasTypedField<EJson::Number>(TEXT("fps")))
	{
		FileFps = FMath::Max(1.0, Root->GetNumberField(TEXT("fps")));
	}
	else if (Root->HasTypedField<EJson::Number>(TEXT("FPS")))
	{
		FileFps = FMath::Max(1.0, Root->GetNumberField(TEXT("FPS")));
	}
	TeacherFrameDt = (FileFps > KINDA_SMALL_NUMBER) ? (1.f / (float)FileFps) : GetModelDt();

	if (bHasFrames)
	{
		TeacherFrames.Reserve(Frames->Num());
		for (int32 idx = 0; idx < Frames->Num(); ++idx)
		{
			const TSharedPtr<FJsonObject> FrameObj = (*Frames)[idx]->AsObject();
			if (!FrameObj.IsValid())
			{
				continue;
			}

			FTeacherFrame Frame;
			if (!BuildRawStateFromFrameJson(FrameObj, Frame.RawState, &Frame.PoseSrc, &Frame.CondRaw))
			{
				UE_LOG(LogTemp, Warning, TEXT("[Teacher] Skip invalid frame %d in %s"), idx, *AbsPath);
				continue;
			}
			if (Frame.RawState.Num() != InputDim)
			{
				Frame.RawState.SetNum(InputDim);
			}
			if (CondDim > 0)
			{
				const int32 TargetDim = FMath::Max(CondDim, 0);
				if (Frame.CondRaw.Num() != TargetDim)
				{
					Frame.CondRaw.SetNumZeroed(TargetDim);
				}
			}
			TeacherFrames.Add(MoveTemp(Frame));
		}
	}
	else
	{
		const TSharedPtr<FJsonObject> TeacherObj = *TeacherObjPtr;
		const TArray<TSharedPtr<FJsonValue>>* StateArray = nullptr;
		const TArray<TSharedPtr<FJsonValue>>* CondArray = nullptr;

	if (!TeacherObj->TryGetArrayField(TEXT("state_norm"), StateArray) || !StateArray || StateArray->Num() == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("[Teacher] %s teacher payload missing state_norm."), *AbsPath);
		return false;
	}
	if (CondDim > 0)
	{
		if (!TeacherObj->TryGetArrayField(TEXT("cond"), CondArray) || !CondArray)
		{
			UE_LOG(LogTemp, Error, TEXT("[Teacher] %s teacher payload missing cond array (CondDim=%d)."), *AbsPath, CondDim);
			return false;
		}
		if (CondArray->Num() != StateArray->Num())
		{
			UE_LOG(LogTemp, Error, TEXT("[Teacher] %s cond/state length mismatch: cond=%d state=%d."), *AbsPath, CondArray->Num(), StateArray->Num());
			return false;
		}
	}

		TeacherFrames.Reserve(StateArray->Num());

		TArray<TArray<float>> AuxContactsRows;
		TArray<TArray<float>> AuxAngVelRows;
		TArray<TArray<float>> AuxPoseHistRows;
		if (TeacherObj->HasTypedField<EJson::Object>(TEXT("aux_inputs")))
		{
			const TSharedPtr<FJsonObject> AuxObj = TeacherObj->GetObjectField(TEXT("aux_inputs"));
			auto ParseAuxMatrix = [](const TArray<TSharedPtr<FJsonValue>>* Arr) -> TArray<TArray<float>>
			{
				TArray<TArray<float>> Result;
				if (!Arr)
				{
					return Result;
				}
				Result.SetNum(Arr->Num());
				for (int32 i = 0; i < Arr->Num(); ++i)
				{
					const TSharedPtr<FJsonValue>& RowVal = (*Arr)[i];
					if (!RowVal.IsValid())
					{
						continue;
					}
					const TArray<TSharedPtr<FJsonValue>> RowArr = RowVal->AsArray();
					Result[i].SetNum(RowArr.Num());
					for (int32 j = 0; j < RowArr.Num(); ++j)
					{
						Result[i][j] = (float)RowArr[j]->AsNumber();
					}
				}
				return Result;
			};

			const TArray<TSharedPtr<FJsonValue>>* AuxContactsJson = nullptr;
			if (AuxObj->TryGetArrayField(TEXT("contacts"), AuxContactsJson) && AuxContactsJson)
			{
				AuxContactsRows = ParseAuxMatrix(AuxContactsJson);
			}
			const TArray<TSharedPtr<FJsonValue>>* AuxAngvelJson = nullptr;
			if (AuxObj->TryGetArrayField(TEXT("angvel_norm"), AuxAngvelJson) && AuxAngvelJson)
			{
				AuxAngVelRows = ParseAuxMatrix(AuxAngvelJson);
			}
			const TArray<TSharedPtr<FJsonValue>>* AuxPoseHistJson = nullptr;
			if (AuxObj->TryGetArrayField(TEXT("pose_hist_norm"), AuxPoseHistJson) && AuxPoseHistJson)
			{
				AuxPoseHistRows = ParseAuxMatrix(AuxPoseHistJson);
			}
		}

		for (int32 idx = 0; idx < StateArray->Num(); ++idx)
		{
			const TSharedPtr<FJsonValue>& RowVal = (*StateArray)[idx];
			const TSharedPtr<FJsonValue>& CondVal = (CondArray && CondArray->IsValidIndex(idx)) ? (*CondArray)[idx] : nullptr;
				if (!RowVal.IsValid() || (CondDim > 0 && !CondVal.IsValid()))
				{
				UE_LOG(LogTemp, Warning, TEXT("[Teacher] Skip invalid frame %d in %s (missing cond/state)."), idx, *AbsPath);
				continue;
			}

			const TArray<TSharedPtr<FJsonValue>> Row = RowVal->AsArray();
			TArray<float> StateNorm;
			StateNorm.Init(0.f, InputDim);
			const int32 CopyCount = FMath::Min(InputDim, Row.Num());
			for (int32 c = 0; c < CopyCount; ++c)
			{
				StateNorm[c] = (float)Row[c]->AsNumber();
			}

			FTeacherFrame Frame;
			DenormX_Z_To_Raw(StateNorm, Frame.RawState);
			BuildPoseFromRawState(Frame.RawState, Frame.PoseSrc);

			if (Frame.RawState.Num() != InputDim)
			{
				Frame.RawState.SetNum(InputDim);
			}
			if (Frame.PoseSrc.Num() != kTrackedBones.Num())
			{
				Frame.PoseSrc.SetNum(kTrackedBones.Num());
				for (FTransform& T : Frame.PoseSrc)
				{
					T = FTransform::Identity;
				}
			}

			if (CondDim > 0)
			{
				const TArray<TSharedPtr<FJsonValue>> CondRow = CondVal->AsArray();
				if (CondRow.Num() < CondDim)
				{
					UE_LOG(LogTemp, Warning, TEXT("[Teacher] Frame %d cond dim mismatch (%d vs expected %d) in %s."), idx, CondRow.Num(), CondDim, *AbsPath);
					continue;
				}
				Frame.CondRaw.Init(0.f, CondDim);
				for (int32 c = 0; c < CondDim; ++c)
				{
					Frame.CondRaw[c] = (float)CondRow[c]->AsNumber();
				}
			}

			if (AuxContactsRows.Num() > 0)
			{
				const int32 useIdx = FMath::Clamp(idx, 0, AuxContactsRows.Num() - 1);
				Frame.ContactsAux = AuxContactsRows[useIdx];
			}
			if (AuxAngVelRows.Num() > 0)
			{
				const int32 useIdx = FMath::Clamp(idx, 0, AuxAngVelRows.Num() - 1);
				Frame.AngVelAux = AuxAngVelRows[useIdx];
			}
			if (AuxPoseHistRows.Num() > 0)
			{
				const int32 useIdx = FMath::Clamp(idx, 0, AuxPoseHistRows.Num() - 1);
				Frame.PoseHistAux = AuxPoseHistRows[useIdx];
			}

			TeacherFrames.Add(MoveTemp(Frame));
		}
	}

	if (TeacherFrames.Num() == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("[Teacher] No valid frames loaded from %s"), *AbsPath);
		return false;
	}

	TeacherFrameCursor = FMath::Clamp(StartupSeedFrameIndex, 0, TeacherFrames.Num() - 1);
	bTeacherClipReady = true;
	UE_LOG(LogTemp, Log, TEXT("[Teacher] Loaded %d frames from %s (fps=%.2f)"), TeacherFrames.Num(), *AbsPath, (float)FileFps);
	return true;
}

bool UEnemyAnimInstance::ShouldUseTeacherPlayback() const
{
	return bEnableTeacherPlayback && bTeacherClipReady && TeacherFrames.Num() > 0;
}

bool UEnemyAnimInstance::AdvanceTeacherClip(bool bPublishPose)
{
	if (!ShouldUseTeacherPlayback())
	{
		return false;
	}

	if (!TeacherFrames.IsValidIndex(TeacherFrameCursor))
	{
		TeacherFrameCursor = bLoopTeacherPlayback ? 0 : FMath::Clamp(TeacherFrameCursor, 0, FMath::Max(TeacherFrames.Num() - 1, 0));
	}

	if (!TeacherFrames.IsValidIndex(TeacherFrameCursor))
	{
		return false;
	}

	const FTeacherFrame& Frame = TeacherFrames[TeacherFrameCursor];
	LastTeacherPoseSrc = Frame.PoseSrc;
	// 仅在非 TeacherForcingEval 时才直接发布 teacher 姿态；teacher-forcing 驱动模型时，姿态由 StepModelFused 发布预测结果
	if (bPublishPose && !IsTeacherModelDriving())
	{
		PoseBox.Publish(Frame.PoseSrc, Frame.RawState, /*bLog=*/false, GFrameNumber);
	}

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
	Prev_X_raw = Frame.RawState;
	NormalizeXRaw_To_Z(Frame.RawState, CurrentMotionStateNorm);
	CarryX_Norm = CurrentMotionStateNorm;
	PushPoseHistoryFrame(Frame.PoseSrc);
	if (CondDim > 0 && Frame.CondRaw.Num() > 0)
	{
		const int32 TargetDim = FMath::Max(CondDim, 0);
		CurrentTeacherCondRaw.SetNum(TargetDim);
		for (int32 i = 0; i < TargetDim; ++i)
		{
			CurrentTeacherCondRaw[i] = Frame.CondRaw.IsValidIndex(i) ? Frame.CondRaw[i] : 0.f;
		}
		bTeacherCondValid = true;

		static int32 DebugTeacherCondLogs = 0;
		if (DebugTeacherCondLogs < 8)
		{
			FString Slice;
			for (int32 idx = 0; idx < TargetDim && idx < 7; ++idx)
			{
				Slice += FString::Printf(TEXT("%s%.4f"), (idx == 0 ? TEXT("") : TEXT(",")), CurrentTeacherCondRaw[idx]);
			}
			UE_LOG(LogTemp, Warning, TEXT("[TDbg][AdvanceTeacherClip] cond[0..6]=%s len=%d target=%d"), *Slice, CurrentTeacherCondRaw.Num(), TargetDim);
			++DebugTeacherCondLogs;
		}
	}
	else
	{
		CurrentTeacherCondRaw.Reset();
		bTeacherCondValid = false;
	}

	if (Frame.ContactsAux.Num() > 0)
	{
		CurrentTeacherContactsNorm = Frame.ContactsAux;
	}
	else
	{
		CurrentTeacherContactsNorm.Reset();
	}
	if (Frame.AngVelAux.Num() > 0)
	{
		CurrentTeacherAngVelNorm = Frame.AngVelAux;
	}
	else
	{
		CurrentTeacherAngVelNorm.Reset();
	}
	if (Frame.PoseHistAux.Num() > 0)
	{
		CurrentTeacherPoseHistNorm = Frame.PoseHistAux;
	}
	else
	{
		CurrentTeacherPoseHistNorm.Reset();
	}
#endif

	TeacherFrameCursor++;
	if (TeacherFrameCursor >= TeacherFrames.Num())
	{
		TeacherFrameCursor = bLoopTeacherPlayback ? 0 : TeacherFrames.Num() - 1;
	}

	return true;
}

void UEnemyAnimInstance::UpdateFootContacts()
{
    ACharacter* Char = Cast<ACharacter>(TryGetPawnOwner());
    USkeletalMeshComponent* SK = GetSkelMeshComponent();
    if (!Char || !SK) return;

    UCharacterMovementComponent* Move = Char->GetCharacterMovement();
    const FFindFloorResult*      FF    = Move ? &Move->CurrentFloor : nullptr;

    // --- 0) 地板平面有效性 ---
    bool bHaveFloor = (FF && FF->bBlockingHit && FF->bWalkableFloor && FF->HitResult.Normal.Z > SMALL_NUMBER);

    // 胶囊半径用于“脚XY是否仍在胶囊可代表的地板上”判断
    const UCapsuleComponent* Capsule = Char->GetCapsuleComponent();
    const float CapsuleR = Capsule ? Capsule->GetScaledCapsuleRadius() : 34.f;

    // --- 1) 脚底世界点 ---
    const FName BoneL(TEXT("ball_l"));
    const FName BoneR(TEXT("ball_r"));

    FVector FootL = SK->GetBoneLocation(BoneL, EBoneSpaces::WorldSpace);
    FVector FootR = SK->GetBoneLocation(BoneR, EBoneSpaces::WorldSpace);

    // 默认 Up（腾空/无地板时兜底）
    FVector N  = FVector::UpVector;
    FVector P0 = Char->GetActorLocation();

    if (bHaveFloor)
    {
        N  = FF->HitResult.ImpactNormal.GetSafeNormal();
        P0 = FF->HitResult.ImpactPoint;
    }

    // 鞋底厚度：把骨点沿法线向下偏移
    FootL -= N * FootSoleOffsetCm;
    FootR -= N * FootSoleOffsetCm;

    // --- 2) 速度（补：Vxy） ---
    const float Dt = FMath::Max(1e-3f, GetWorld() ? GetWorld()->GetDeltaSeconds() : 1.f/60.f);
    float VzL = 0.f, VzR = 0.f, VxyL = 0.f, VxyR = 0.f;
    if (bPrevFeetValid)
    {
        const FVector dL = FootL - PrevFootL_WS;
        const FVector dR = FootR - PrevFootR_WS;
        VzL  = FMath::Abs(dL.Z / Dt);
        VzR  = FMath::Abs(dR.Z / Dt);
        VxyL = FVector2D(dL.X, dL.Y).Size() / Dt;   // ★ 新增：水平速度 cm/s
        VxyR = FVector2D(dR.X, dR.Y).Size() / Dt;   // ★ 新增
    }
    PrevFootL_WS = FootL;
    PrevFootR_WS = FootR;
    bPrevFeetValid = true;

    // --- 3) 脚到地板平面的距离 d = (P - P0)·N ---
    auto DistToPlane = [&](const FVector& P){ return FVector::DotProduct(P - P0, N); };
    float dL = DistToPlane(FootL);
    float dR = DistToPlane(FootR);

    // --- 4) XY 合法性 ---
    const FVector ActorXY(Char->GetActorLocation().X, Char->GetActorLocation().Y, 0.f);
    auto XYTooFar = [&](const FVector& P){
        const FVector Pxy(P.X, P.Y, 0.f);
        return (Pxy - ActorXY).Size() > (CapsuleR + MaxFootRadialFromCapsule);
    };
    const bool bL_XY_OK = !XYTooFar(FootL);
    const bool bR_XY_OK = !XYTooFar(FootR);

    // --- 5) 进入/退出条件 + 最小时长（对齐提取端语义） ---
    auto Enter = [&](float d, float Vz, float Vxy){
        return (d <= ContactEnterCm) && (Vz <= EnterVzMaxCmPs) && (Vxy <= EnterVxyMaxCmPs);
    };
    auto Exit = [&](float d, float Vz, float Vxy){
        return (d > ContactExitCm) || (Vz >= ExitVzMinCmPs) || (Vxy >= ExitVxyMinCmPs);
    };

    auto TickAccum = [&](bool bOld, bool bNew, float& AccOn, float& AccOff){
        if (bNew) { AccOn += Dt;  AccOff = 0.f; }
        else      { AccOff += Dt; AccOn  = 0.f; }
    };

    auto UpdateOne = [&](bool bOld, float d, float Vz, float Vxy, bool bXYok, float& AccOn, float& AccOff)->bool
    {
        if (!bHaveFloor || !bXYok) return bOld; // 地板/XY不靠谱：保持
        bool bNew = bOld;
        if (!bOld) { if (Enter(d,Vz,Vxy)) bNew = true; }
        else       { if (Exit (d,Vz,Vxy)) bNew = false; }

        // 最小 on/off 时长（秒）
        TickAccum(bOld, bNew, AccOn, AccOff);
        if (!bOld && bNew && AccOn   < MinOnTimeSec)  bNew = false;
        if ( bOld && !bNew && AccOff < MinOffTimeSec) bNew = true;
        return bNew;
    };

    bool NewL = UpdateOne(bContactLeft,  dL, VzL, VxyL, bL_XY_OK, AccOnL, AccOffL);
    bool NewR = UpdateOne(bContactRight, dR, VzR, VxyR, bR_XY_OK, AccOnR, AccOffR);
    bContactLeft  = NewL;
    bContactRight = NewR;

    // --- 6) 写回到 X 的 Contacts 切片（按 mu/std 归一化） ---
    if (const FStateSlice* S = StateLayout.Find(TEXT("Contacts")))
    {
        auto WriteNorm = [&](int idx, float raw){
            if (!CarryX_Norm.IsValidIndex(idx)) return;
            const float mu  = (MuX.IsValidIndex(idx)  ? MuX[idx]  : 0.f);
            const float std = (StdX.IsValidIndex(idx) ? StdX[idx] : 1.f);
            CarryX_Norm[idx] = (raw - mu) / FMath::Max(std, 1e-6f);
        };
        WriteNorm(S->Start + 0, bContactLeft  ? 1.f : 0.f);
        WriteNorm(S->Start + 1, bContactRight ? 1.f : 0.f);
    }
}

const FStateSlice* UEnemyAnimInstance::FindSliceChecked(const TMap<FString, FStateSlice>& InLayout, const TCHAR* Name)
{
	if (const FStateSlice* S = InLayout.Find(Name)) return S;
	UE_LOG(LogTemp, Error, TEXT("[ApplyRoot] Missing slice: %s"), Name);
	return nullptr;
}




#if ENEMY_ANIM_WITH_ONNX_RUNTIME
void UEnemyAnimInstance::BuildCondVector(TArray<float>& OutCond)
{
	const int32 Dim = (CondDim > 0) ? CondDim : 0;
	if (Dim <= 0)
	{
		OutCond.Reset();
		return;
	}

	TArray<float> RawCond;
	RawCond.Init(0.f, Dim);

	const bool bUseTeacherCond = IsTeacherModelDriving() && bTeacherCondValid && CurrentTeacherCondRaw.Num() >= Dim;
	if (bUseTeacherCond)
	{
		for (int32 i = 0; i < Dim; ++i)
		{
			RawCond[i] = CurrentTeacherCondRaw.IsValidIndex(i) ? CurrentTeacherCondRaw[i] : 0.f;
		}
	}
	else
	{
		auto WriteIfValid = [&](int32 Index, float Value)
		{
			if (RawCond.IsValidIndex(Index))
			{
				RawCond[Index] = Value;
			}
		};

		// 动作 one-hot
		WriteIfValid(0, FMath::Clamp(ActionBits.X, 0.f, 1.f));
		WriteIfValid(1, FMath::Clamp(ActionBits.Y, 0.f, 1.f));
		WriteIfValid(2, FMath::Clamp(ActionBits.Z, 0.f, 1.f));
		WriteIfValid(3, FMath::Clamp(ActionBits.W, 0.f, 1.f));

		// 方向与速度（米/秒）
		float SpeedMs = 0.f;
		FVector2D Dir = ResolvePlanarDirection(SpeedMs);
		Dir.X = FMath::Clamp(Dir.X, -1.f, 1.f);
		Dir.Y = FMath::Clamp(Dir.Y, -1.f, 1.f);

		WriteIfValid(4, Dir.X);
		WriteIfValid(5, Dir.Y);
		WriteIfValid(6, SpeedMs);
	}

	PushCondHistorySample(RawCond);
	ApplyCondNormalization(RawCond, OutCond);

	static int32 DebugCondBuildLogs = 0;
	if (DebugCondBuildLogs < 8)
	{
		FString RawSlice, OutSlice;
		for (int32 i = 0; i < Dim && i < 7; ++i)
		{
			RawSlice += FString::Printf(TEXT("%s%.4f"), (i == 0 ? TEXT("") : TEXT(",")), RawCond.IsValidIndex(i) ? RawCond[i] : 0.f);
			OutSlice += FString::Printf(TEXT("%s%.4f"), (i == 0 ? TEXT("") : TEXT(",")), OutCond.IsValidIndex(i) ? OutCond[i] : 0.f);
		}
		UE_LOG(LogTemp, Warning, TEXT("[TDbg][BuildCondVector] useTeacher=%d Dim=%d Raw=%s Out=%s"),
			(int32)bUseTeacherCond, Dim, *RawSlice, *OutSlice);
		++DebugCondBuildLogs;
	}
}
#endif



void UEnemyAnimInstance::FPoseMailbox::Publish(const TArray<FTransform>& InLocals, const TArray<float>& InStateDenorm, bool bLog, uint32 TagFrame)
{
	const int32 write = 1 - ReadSlot.Load();
	Buf[write].Locals = InLocals;
	Buf[write].StateDenorm = InStateDenorm; // <--- [新增] 拷贝状态数据

	const uint32 newSeq = PublishedSeq.Load() + 1;
	Buf[write].Seq     = newSeq;
	Buf[write].TimeSec = FPlatformTime::Seconds();

	PublishedSeq.Store(newSeq);
	ReadSlot.Store(write);

	UE_LOG(LogTemp, Display, TEXT("D: >>> Pose PUBLISHED! Seq=%u, Frame=%u"), newSeq, TagFrame);
}

bool UEnemyAnimInstance::FPoseMailbox::Consume(TArray<FTransform>& OutLocals, TArray<float>& OutStateDenorm, uint32& OutSeq, bool bLog) const
{
	const int32 slot = ReadSlot.Load();
	const uint32 seq0 = Buf[slot].Seq;
	if (seq0 == 0)
	{
		LOGSYNC(bLog, "[Consume] EMPTY (seq=0) readSlot=%d", slot);
		return false;
	}

	OutLocals = Buf[slot].Locals;
	OutStateDenorm = Buf[slot].StateDenorm; // <--- [新增] 拷贝状态数据
	
	const uint32 seq1 = Buf[slot].Seq;
	if (seq0 != seq1)
	{
		LOGSYNC(bLog, "[Consume] RACE seq0=%u seq1=%u readSlot=%d -> drop", seq0, seq1, slot);
		return false;
	}

	OutSeq = seq1;
	LOGSYNC(bLog, "[Consume] seq=%u readSlot=%d locals=%d state_floats=%d", seq1, slot, OutLocals.Num(), OutStateDenorm.Num());
	return true;
}
