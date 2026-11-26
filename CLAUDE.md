@AGENTS.md

# Phần riêng cho Claude Code

Quy tắc chung (ràng buộc, quy trình, tier, eval gate) nằm ở `AGENTS.md` phía trên. File này chỉ ánh xạ chúng sang tool của Claude Code.

## Model cho sub-agent

Giao việc qua `/orchestration` (worker Orca) là mặc định; tool `Agent` chỉ dùng khi Orca không chạy được. Luôn truyền model và effort theo bảng: worker Orca dùng `--model <ID>` và `--effort`, tool `Agent` dùng `model` và `effort`.

| Vai trò | model (ID cho Orca) | effort |
|---|---|---|
| Tier Dễ | haiku (`claude-haiku-5-5`) | mặc định |
| Tier Trung bình | sonnet (`claude-sonnet-5-5`) | high |
| Tier Khó | sonnet (`claude-sonnet-5-5`) | high |
| Reviewer từng task | opus (`claude-opus-5-5`) | mặc định |
| Review cuối toàn branch, trước khi mở PR hoặc merge | opus (`claude-opus-5-5`) | high |

Hai quy tắc dưới đây ghi đè `~/.claude/CLAUDE.md` riêng cho dự án này:

- Opus dùng cho reviewer từng task và review cuối, không cần người dùng yêu cầu lại trong phiên.
- Haiku chỉ làm tier Dễ. Mọi review, kể cả re-review diff nhỏ, giao cho opus hoặc người điều phối.

Trung bình và Khó dùng cùng model, khác nhau ở bước thiết kế. Vì vậy khi nâng tier từ Trung bình lên Khó, phải viết brief mới có bước thiết kế; gửi lại cùng brief cho cùng model là vô ích.

## Song song

"Worktree riêng" trong `AGENTS.md` nghĩa là worker Orca với `--worktree new-child`, hoặc `isolation: "worktree"` khi gọi `Agent`.

Worker trong worktree không chạy `uv run`, `uv sync`, `uv add`, `uv lock` hay `pip`, vì uv sẽ dựng venv mới (kèm torch) cho từng worktree. Worker gọi venv của checkout chính ở chế độ chỉ đọc: `<checkout chính>/.venv/Scripts/python.exe -m pytest -q`, `<checkout chính>/.venv/Scripts/ruff.exe`.
