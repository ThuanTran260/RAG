@AGENTS.md

# Phần riêng cho Claude Code

Quy tắc chung (ràng buộc, quy trình, tier, eval gate) nằm ở `AGENTS.md` phía trên. File này chỉ ánh xạ chúng sang tool của Claude Code.

## Model cho sub-agent

Khi gọi `Agent`, luôn truyền `model` và `effort` theo bảng.

| Tier | model | effort |
|---|---|---|
| Dễ | haiku | mặc định |
| Trung bình | sonnet | high |
| Khó | sonnet | high |
| Reviewer | opus | mặc định |

Hai quy tắc dưới đây ghi đè `~/.claude/CLAUDE.md` riêng cho dự án này:

- Opus chỉ dùng cho reviewer, và được dùng mà không cần người dùng yêu cầu lại trong phiên.
- Haiku chỉ làm tier Dễ. Mọi review, kể cả re-review diff nhỏ, giao cho opus hoặc người điều phối.

Trung bình và Khó dùng cùng model, khác nhau ở bước thiết kế. Vì vậy khi nâng tier từ Trung bình lên Khó, phải viết brief mới có bước thiết kế; gửi lại cùng brief cho cùng model là vô ích.

## Song song

"Worktree riêng" trong `AGENTS.md` nghĩa là truyền `isolation: "worktree"` khi gọi `Agent`.
