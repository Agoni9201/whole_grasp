#!/usr/bin/env python3
"""whole-grasp 主程序入口（第一步：抓取阶段）。

等价于原 real-nero-grasp 工程的 real_nero_single_grasp_runner.py：
所有 CLI 参数、主流程都在 real_nero_grasp.runtime 里。
"""

# 运行方式（两种等价）：

# ```bash
# python main.py --calib-json <标定.json> [其他参数]
# python -m real_nero_grasp --calib-json <标定.json> [其他参数]   # 包自带的 __main__.py
# ```

from real_nero_grasp import runtime as _runtime

if __name__ == "__main__":
    raise SystemExit(_runtime.main())