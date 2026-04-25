# Docs 模板目录

这个目录按**目标文档目录**分层，而不是把所有模板平铺在一层。

## 目录约定

| 模板目录 | 目标落点 | 用途 |
|---|---|---|
| `docs/templates/changes/` | `docs/changes/` | 活跃改动、重构计划、Phase 报告、验证记录、最终 change report |
| `docs/templates/delete/` | `docs/delete/` | 删除审计、dead-code inventory、compat/legacy cleanup 清单 |

## 使用规则

1. 新模板必须放到与目标文档目录对应的子目录。
2. `docs/changes/` 与 `docs/delete/` 只放已经实例化的任务文档，不再放空模板。
3. 模板文件名描述文档目的，例如 `change_report_template.md`、`delete_audit_template.md`。
4. 如果某个 docs 子目录开始出现重复结构，先在 `docs/templates/<target>/` 下补模板，再写新文档。
5. 历史模板已从 `docs/templates/*.md` 收敛到 `docs/templates/changes/`，避免根目录继续变平。

