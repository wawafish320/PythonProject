// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include <malloc/_platform.h>

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "Animation/AnimInstanceProxy.h"

#ifndef ENEMY_ANIM_WITH_ONNX_RUNTIME
#define ENEMY_ANIM_WITH_ONNX_RUNTIME 1
#endif

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
#include "Templates/UniquePtr.h"
#include "NNE.h"                  // 定义 IModel 接口
#include "NNERuntime.h"           // 定义 INNERuntime 接口
#include "onnxruntime_cxx_api.h"
#endif



#include "EnemyAnimInstance.generated.h"

class FJsonObject;


#define LOGSYNC(Cond, Fmt, ...) \
do { if (Cond) { UE_LOG(LogTemp, Warning, TEXT(Fmt), ##__VA_ARGS__); } } while(0)

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
namespace Ort
{
	struct Env;
}

namespace UE::NNE
{
	class IModelInstanceCPU;
	class IModelCPU;
}
#endif

class UNNEModelData;
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
class FFusedEventMotionModel;
struct FFusedEventMotionModelDeleter
{
	void operator()(FFusedEventMotionModel* Ptr) const;
};

struct FFusedModelInputDims
{
	int32 Motion = 0;
	int32 Cond = 0;
	int32 Contacts = 0;
	int32 AngVel = 0;
	int32 PoseHist = 0;
};
#endif
class USkeletalMesh;


static FORCEINLINE FString V2(const FVector& v){ return FString::Printf(TEXT("(%.3f,%.3f,%.3f)"), v.X,v.Y,v.Z); }
static FORCEINLINE FString V2_2D(const FVector& v){ return FString::Printf(TEXT("(%.3f,%.3f)"), v.X,v.Y); }



// ---------------------- 自定义 Proxy：跨线程传 X/Gen -------------------------
struct FEnemyAnimInstanceProxy : public FAnimInstanceProxy
{
	TArray<float> XNorm;
	uint64 Gen = 0;
	FEnemyAnimInstanceProxy(UAnimInstance* InAnimInstance) : FAnimInstanceProxy(InAnimInstance) {}
	void PushFromGT(const TArray<float>& In, uint64 NewGen){ XNorm = In; Gen = NewGen; }
	const TArray<float>& PullForWT(uint64& OutGen) const { OutGen = Gen; return XNorm; }

	TArray<FTransform> Payload_LocalTracked;
	TArray<FName>      Payload_BoneNames;
	bool bPayloadIsLocal = false, bHasNewPayload = false;
	FCriticalSection   PayloadCS;

	FCriticalSection PoseLock;

	void PushPoseLocalFromGT(const TArray<FTransform>& LocalTracked, const TArray<FName>& UseBones);

	// WT：真正把姿势写进 Output
	virtual auto Evaluate(FPoseContext& Output) -> bool override;
};



UENUM(BlueprintType)
enum class ESixDMode : uint8
{
	XZ,
	ZX,
	XY,
	YX,
	ZY,
	YZ,

};





USTRUCT()
struct FSixDDecodeSpec {
	GENERATED_BODY()
	UPROPERTY() ESixDMode Mode = ESixDMode::XZ; // XY 还是 XZ
	UPROPERTY() bool bSwapC0C1 = false;
	UPROPERTY() uint8 NegMask = 0;  // bit0=negC0, bit1=negC1
	UPROPERTY() uint8 Permute = 0;  // 0..5: XYZ,YZX,ZXY,XZY,YXZ,ZYX
	bool operator==(const FSixDDecodeSpec& o) const {
		return Mode==o.Mode && bSwapC0C1==o.bSwapC0C1 && NegMask==o.NegMask && Permute==o.Permute;
	}

	bool IsValid() const {
		return (Mode==ESixDMode::XY || Mode==ESixDMode::XZ) && (Permute <= 5);
	}
};

// [新增] 用于定义条件向量C的归一化模式
UENUM(BlueprintType)
enum class ECondNormMode : uint8
{
	// 不对C向量做任何处理，直接使用原始值
	Raw,
	
	ZScore
};



USTRUCT(BlueprintType)
struct FStateSlice
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	int32 Start = 0;

	UPROPERTY(EditAnywhere, BlueprintReadOnly)
	int32 Size = 0;
};

// 精简的骨骼拓扑缓存（供姿态写回）
struct FRefCache
{
	TArray<int32>   TrackedParent;
	TArray<FVector> OffInParentLocal;
	TArray<int32>   TopoOrder;

	const USkeletalMesh* BoundMesh = nullptr;
	bool bOk = false;
};


/**
 * 
 */
UCLASS()
class TEST_API UEnemyAnimInstance : public UAnimInstance
{
	GENERATED_BODY()

public:
	virtual ~UEnemyAnimInstance();




	// 供GT发布姿态：模型输出的“相对根”的局部变换（仍是源坐标系）
	struct FPoseMailbox
    {
        struct FSlot
        {
            TArray<FTransform> Locals;
        	// 用于传递完整的反归一化状态向量
        	TArray<float> StateDenorm; 
            uint32 Seq = 0;
            double TimeSec = 0.0;
        };
    
        FSlot Buf[2];
        TAtomic<int32>  ReadSlot{0};     // 当前可读槽 [0|1]
        TAtomic<uint32> PublishedSeq{0}; // 已发布序号
    
		void Publish(const TArray<FTransform>& InLocals, const TArray<float>& InStateDenorm, bool bLog=false, uint32 TagFrame=0);
		bool Consume(TArray<FTransform>& OutLocals, TArray<float>& OutStateDenorm, uint32& OutSeq, bool bLog=false) const;
    };
	

	/** schema_arpg_focused.json（导出时生成） */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion")
	FFilePath SchemaJsonPath;

	// === Startup seed (optional): use first frame from a JSON clip ===
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion|Startup")
	FFilePath StartupSeedJson;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion|Startup", meta=(ClampMin="0"))
	int32 StartupSeedFrameIndex = 0;

	// === Foot-contact config (aligned with extractor) ===
	UPROPERTY(EditAnywhere, Category="Contacts") float FootSoleOffsetCm        = 2.0f;  // 与提取SweepUpOffset≈2一致
	UPROPERTY(EditAnywhere, Category="Contacts") float ContactEnterCm          = 2.5f;  // 进入阈值 (提取=2.5)
	UPROPERTY(EditAnywhere, Category="Contacts") float ContactExitCm           = 5.0f;  // 退出阈值 (提取=5.0)

	// 分离“进入/退出”的速度门限（提取端是两套值）
	UPROPERTY(EditAnywhere, Category="Contacts") float EnterVzMaxCmPs          = 5.0f;   // 进入：|Vz| < 5
	UPROPERTY(EditAnywhere, Category="Contacts") float ExitVzMinCmPs           = 15.0f;  // 退出：Vz > +15
	UPROPERTY(EditAnywhere, Category="Contacts") float EnterVxyMaxCmPs         = 12.0f;  // 进入：||Vxy|| < 12
	UPROPERTY(EditAnywhere, Category="Contacts") float ExitVxyMinCmPs          = 18.0f;  // 退出：||Vxy|| > 18

	// 以“秒”为单位表达最小 On/Off 时长（提取=3 帧@60Hz≈0.05s）
	UPROPERTY(EditAnywhere, Category="Contacts") float MinOnTimeSec            = 0.05f;
	UPROPERTY(EditAnywhere, Category="Contacts") float MinOffTimeSec           = 0.05f;

	// 额外保护（可留着）：脚XY偏离胶囊太远时不改变状态
	UPROPERTY(EditAnywhere, Category="Contacts") float MaxFootRadialFromCapsule= 40.f;

	// 运行时缓存
	FVector PrevFootL_WS = FVector::ZeroVector;
	FVector PrevFootR_WS = FVector::ZeroVector;
	bool    bPrevFeetValid = false;
	float   AccOnL = 0.f, AccOffL = 0.f;   // 左脚累计 on/off 时间（秒）
	float   AccOnR = 0.f, AccOffR = 0.f;   // 右脚累计 on/off 时间（秒）
	bool    bContactLeft  = false;
	bool    bContactRight = false;

    
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "AI Motion|History")
	TArray<FTransform> PoseHistoryBuffer;

#if ENEMY_ANIM_WITH_ONNX_RUNTIME
		TUniquePtr<FFusedEventMotionModel, FFusedEventMotionModelDeleter> NeuralNetwork;
#endif



	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Motion Control | Wander")
	bool bEnableWanderOffset = true;

	/** 漫步时，与正前方偏离的最大角度（单位：度）*/
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Motion Control | Wander", meta = (EditCondition = "bEnableWanderOffset"))
	float MaxWanderAngle = 45.0f;

	/** 漫步时，左右摆动的速度（值越大，摆动越快）*/
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "AI Motion Control | Wander", meta = (EditCondition = "bEnableWanderOffset"))
	float WanderSpeed = 0.5f;

	/** 是否直接使用离线 JSON 帧驱动（Teacher 模式），跳过模型滚动 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion|Teacher")
	bool bEnableTeacherPlayback = true;

	/** Teacher JSON 的路径（为空则回退到 StartupSeedJson） */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion|Teacher", meta=(EditCondition="bEnableTeacherPlayback"))
	FFilePath TeacherClipJsonPath;

	/** Teacher 序列播放完是否循环 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion|Teacher", meta=(EditCondition="bEnableTeacherPlayback"))
	bool bLoopTeacherPlayback = true;

	/** Teacher 播放但仍使用模型推理（Teacher forcing：喂真值 X/C，只看模型输出 Y） */
	UPROPERTY()
	bool bUseTeacherForcingEval = true;


	float TrajHz = 60.f;
	
	float ModelDt = 1.f / 60.f; 

	FORCEINLINE float GetModelDt() const
	{
		return (TrajHz > 0.f) ? (1.f / TrajHz) : (1.f / 60.f);
	}

protected:
	// 解析 schema/csv
	bool LoadSchemaAndStats();


	// ========== UEnemyAnimInstance::ResetARState ==========
	// 作用：避免“零输入陷阱”。构造一个合理的 X_raw（以 MuX 为基底，叠加当前姿态），
	//      然后用 MuX/StdX 白化到 CurrentMotionStateNorm，清空循环态。
#if ENEMY_ANIM_WITH_ONNX_RUNTIME
	void ResetARState();
	
	void LoadOnnxModel();
	void EnsurePoseHistoryConfig(int32 RequiredPoseHistDim);
	FFusedModelInputDims BuildModelInputDims() const;

	// 单步推理（自回归）：x_{t+1} = f(x_t, cond_t)
	bool StepModel(float DeltaSeconds);
	bool StepModelFused(float DeltaSeconds);

	void BuildCondVector(TArray<float>& OutCond);
#else
	void LoadOnnxModel() {}
#endif


private:
	// --- PATCH: Proxy 接入 ---
	virtual FAnimInstanceProxy* CreateAnimInstanceProxy() override;
	virtual void DestroyAnimInstanceProxy(FAnimInstanceProxy* InProxy) override;
	FORCEINLINE FEnemyAnimInstanceProxy& ProxyGT(){ return GetProxyOnGameThread<FEnemyAnimInstanceProxy>(); }
	FORCEINLINE FEnemyAnimInstanceProxy& ProxyAny(){ return GetProxyOnAnyThread<FEnemyAnimInstanceProxy>(); }

	// --- PATCH: 每帧把 GT 的状态推到 Proxy ---
	virtual void PreUpdateAnimation(float DeltaSeconds) override;
	// --- PATCH: GT 侧状态代号（是否拿到新状态） ---
	UPROPERTY(Transient) uint64 GT_StateGen = 0;
	
protected:
	

	// 用于在编辑器中指定ONNX模型资产
	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "ONNX Motion")
	TObjectPtr<UNNEModelData> OnnxModelDataAsset;

	UPROPERTY(Transient) FSixDDecodeSpec DecodeSpec;

	UPROPERTY(Transient) TArray<FTransform> PredictedLocal_Src;
	UPROPERTY(Transient) TArray<FTransform> PredictedLocal_UE;
	UPROPERTY(Transient) bool bPelvisFacingAligned = false;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Coordinate System")
	FQuat CachedQ_SrcToUE = FQuat::Identity; // 将模型“源坐标系”向UE坐标系对齐（首帧测定）

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Coordinate System")
	FQuat CachedQ_UEToSrc = FQuat::Identity; // UE -> 源

	TArray<float> XNorm;


	// 预测出的骨骼局部姿态（根相对）
	TArray<FTransform> PredictedLocal; // size = B

	// 刷新频率（和帧同步）；如果你想用固定 dt（比如采样 60Hz），可以强制使用该 dt
	float AccumulatedTime = 0.f;
	

	// 帧内复用的临时数组（避免每帧 Init/分配）
	FRefCache RefCache;

	TArray<FQuat>   FinalRotsCS;

	TArray<FVector> FinalPossCS;


	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion|Norm")
	FFilePath CondNormJsonPath; // 手动指定 cond_norm.json（可选，优先级最高）

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion|Cond")
	ECondNormMode CondNormMode = ECondNormMode::Raw;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="AI Motion|Cond", meta=(ClampMin="8", ClampMax="240"))
	int32 CondNormWindow = 60;

private:
	
	// ===== 同步/日志控制 =====
	FPoseMailbox PoseBox;
	uint32 LastSeqApplied = 0; 
	
	// 动作向量 (one-hot: idle, walk, run, jump)
	
	FVector4 ActionBits = FVector4(1, 0, 0, 0);

	TArray<float>    CondHistory;
	int32            CondHistoryCursor = 0;
	int32            CondHistoryCount  = 0;

	TArray<float> CurrentTeacherContactsNorm;
	TArray<float> CurrentTeacherAngVelNorm;
	TArray<float> CurrentTeacherPoseHistNorm;

	
	const TArray<FName> kTrackedBones = {
    		TEXT("pelvis"), TEXT("Bip001"), TEXT("spine_01"), TEXT("spine_02"),
    		TEXT("clavicle_l"), TEXT("upperarm_l"), TEXT("RUpArmTwist_l_01"), TEXT("RUpArmTwist_l_02"), 
    		TEXT("lowerarm_l"), TEXT("L_ForeTwist_01"), TEXT("L_ForeTwist_02"),  TEXT("hand_l"),
    		TEXT("index_01_l"), TEXT("middle_01_l"), TEXT("pinky_01_l"),  TEXT("ring_01_l"), TEXT("thumb_01_l"), 
    		TEXT("clavicle_r"), TEXT("upperarm_r"), TEXT("RUpArmTwist_r_01"), TEXT("RUpArmTwist_r_02"), 
    		TEXT("lowerarm_r"), TEXT("R_ForeTwist_01"), TEXT("R_ForeTwist_02"), TEXT("hand_r"),
    		TEXT("index_01_r"), TEXT("middle_01_r"), TEXT("pinky_01_r"),  TEXT("ring_01_r"), TEXT("thumb_01_r"), 
    		TEXT("neck_01"), TEXT("head"),
    		TEXT("thigh_l"), TEXT("calf_l"), TEXT("LCalfTwist_01"), TEXT("LCalfTwist_02"), TEXT("LCalfTwist_03"),
    		TEXT("foot_l"), TEXT("ball_l"),
    		TEXT("thigh_r"), TEXT("calf_r"), TEXT("RCalfTwist_01"), TEXT("RCalfTwist_02"), TEXT("RCalfTwist_03"),
    		TEXT("foot_r"), TEXT("ball_r"),  
    	};          


	// [新增] GT状态脏标记，为true时才在下一帧PreUpdate中推送状态X
	bool bGTStateDirty = false;

	// [新增] 用于标记是否是第一次从GT推送初始状态
	bool bIsFirstGTUpdate = true;


	// [修改] 将 X 的归一化状态向量重命
	TArray<float> CurrentMotionStateNorm;

	// Teacher forcing 临时条件向量（原始域）
	UPROPERTY(Transient)
	TArray<float> CurrentTeacherCondRaw;
	UPROPERTY(Transient)
	bool bTeacherCondValid = false;

	// 上一帧“原始域”的状态向量（训练同构：与 StateLayout 对齐）
	UPROPERTY(Transient)
	TArray<float> Prev_X_raw;

	// teacher 片段最近一帧的姿态（用于推理失败时兜底）
	UPROPERTY(Transient)
	TArray<FTransform> LastTeacherPoseSrc;


	/** 从 y_rows.json 解析的 drop 掩码（长度=OutputDim） */
	TArray<uint8> DropY;

	// ==== 内部状态变量 ====

	// -- 维度信息 --
	int32 InputDim = 0;
	int32 OutputDim = 0;
	int32 CondDim = 7;
    
	// -- 布局信息 --
	TMap<FString, FStateSlice> StateLayout;
	TMap<FString, FStateSlice> OutputLayout;

	// -- 归一化统计数据 (X: State, Y: Output, C: Condition) --
	TArray<float> MuX, StdX;
	TArray<float> MuY, StdY;
	// —— 归一化附加参数（训练端一致）——
	// 注意：尺寸应分别等于 StateLayout["RootVelocity"].Size 和 StateLayout["BoneAngularVelocities"].Size
	TArray<float> TanhScalesRootVel;
	TArray<float> TanhScalesAngVel;
	
	// —— 自回归携带态（GT 侧持有）——
	TArray<float> CarryX_Norm;   // 上一帧输入 X（归一化）

	// Pose history stats
	int32 PoseHistoryLen = 0;
	int32 PoseHistoryHead = 0;
	int32 PoseHistoryValidFrames = 0;
	TArray<float> PoseHistoryScales;
	TArray<float> PoseHistoryMu;
	TArray<float> PoseHistoryStd;

	TArray<FQuat>   AngVelPrevQ;

	bool bSkipBindApplyOnce = false;
	bool bBindReady = false;

	bool bConjugateS2U=false;

	struct FTeacherFrame
	{
		TArray<float> RawState;
		TArray<FTransform> PoseSrc;
		TArray<float> CondRaw;
		TArray<float> ContactsAux;
		TArray<float> AngVelAux;
		TArray<float> PoseHistAux;
	};


	TArray<FTeacherFrame> TeacherFrames;

	UPROPERTY(Transient)
	float TeacherFrameDt = 1.f / 60.f;

	UPROPERTY(Transient)
	int32 TeacherFrameCursor = 0;

	UPROPERTY(Transient)
	bool bTeacherClipReady = false;


	
public:
	
	
	// == AnimInstance 生命周期 ==
	virtual void NativeInitializeAnimation() override;
	
	virtual void NativeUpdateAnimation(float DeltaSeconds) override;

	virtual void NativePostEvaluateAnimation() override;

protected:


	
private:	
	void WarmStartPoseInRawState(TArray<float>& InOutRawState);
	
	
	void UpdateFootContacts();

	void InitializePoseHistory(int32 InHistoryLen);
	
	void ResetPoseHistoryFromPose(const TArray<FTransform>& PoseSrcSpace);
	void PushPoseHistoryFrame(const TArray<FTransform>& PoseSrcSpace);
	void BuildPoseHistoryFeature(TArray<float>& OutFeature) const;
	void NormalizePoseHistoryRaw(const TArray<float>& Raw, TArray<float>& OutNorm) const;
	bool ExtractStateSlice(const FStateSlice* Slice, const TArray<float>& Source, TArray<float>& Out) const;
	bool TryBuildRawStateFromSeedJson(TArray<float>& OutXRaw, TArray<FTransform>* OutLocalUE = nullptr) const;
	bool BuildRawStateFromFrameJson(const TSharedPtr<FJsonObject>& FrameObj, TArray<float>& OutRaw, TArray<FTransform>* OutPoseSrc = nullptr, TArray<float>* OutCondRaw = nullptr) const;
	bool LoadTeacherClip();
	bool AdvanceTeacherClip(bool bPublishPose = true);
	bool ShouldUseTeacherPlayback() const;
	FString ResolveTeacherClipPath() const;
	bool ExtractCondFromFrameJson(const TSharedPtr<FJsonObject>& FrameObj, TArray<float>& OutCond) const;



	const FStateSlice* FindSliceChecked(const TMap<FString, FStateSlice>& InLayout, const TCHAR* Name);


	// —— 归一化 / 反归一化 ——
	// X_raw -> Z   （先全局 μ/σ，再对特殊通道按训练端口径覆盖）
	void NormalizeXRaw_To_Z(const TArray<float>& X_raw, TArray<float>& OutZ) const;
	void DenormX_Z_To_Raw(const TArray<float>& X_norm, TArray<float>& OutRaw) const;
	
	// Y_norm -> Y_raw （先全局 μ/σ 反变换，若 Y 有特殊通道则做逆变换）
	void DenormY_Z_To_Raw(const TArray<float>& Y_norm, TArray<float>& OutRaw) const;

	float AtanhSafe(float x) const;
	void BuildPoseFromRawState(const TArray<float>& RawState, TArray<FTransform>& OutPoseSrc) const;
	void ResetCondHistory();
	void PushCondHistorySample(const TArray<float>& Sample);
	bool ComputeCondWindowStats(TArray<float>& OutMu, TArray<float>& OutStd) const;
	void ApplyCondNormalization(const TArray<float>& RawCond, TArray<float>& OutCond);
	bool IsTeacherModelDriving() const;
	void PublishTeacherFallbackPose();

	FVector2D ResolvePlanarDirection(float& OutSpeedMs) const;

	FORCEINLINE FQuat SafeQ(const FQuat& Q) const
	{
		// 非法值直接返回单位四元数，避免后续 Normalize 产生 NaN
		if (Q.ContainsNaN())
			return FQuat::Identity;

		// 归一化；非常小的长度直接返回单位四元数
		const float SS = Q.SizeSquared();
		if (SS < KINDA_SMALL_NUMBER)
			return FQuat::Identity;

		if (!Q.IsNormalized())
		{
			FQuat R = Q;
			R.Normalize();
			return R;
		}
		return Q;
	}
	

};
