# Oh Deer 年龄分类：上传 GitHub 前先看

**这是适合上传到团队 GitHub 的代码文件夹**，包含训练、预测、画图、错误分析和未来检测框整合程序。

1. 把 ZIP 解压得到的 `OhDeer_Age_GitHub_Ready` 文件夹上传到 GitHub 仓库对应目录，或将文件拷贝到团队现有仓库的 `age_classification/` 中。
2. **不要**上传 `snapshot-wi-oh-deer/`（图片）、原始/清理 CSV、`.venv/` 或 `.pt` 模型。以上文件默认已在 `.gitignore` 中排除。
3. 训练好的两个模型仍然在你电脑的 `age_output_clean/` 内。若队友需要运行分类器，单独协商模型交付和数据许可，不要以为本压缩包已经包含 checkpoint。
4. `README.md` 包含英文介绍、所有 PowerShell 命令、训练配置、报告指标以及 YOLO/MegaDetector 接口。
5. 与老师和队友说明：目前实际完成的是 Full Image 与 **人工标注 GT Crop** 两组实验；**Detected Crop** 尚未完成。
6. 代码主要来自之前用于本次实验的脚本；`make_clean_csv.py` 是这次专为复现清理步骤补充的脚本，不代表当时已运行过这个新文件。

**推荐 GitHub 上传文件：** 根目录七个 `.py` + `requirements.txt` + `README.md` + `README_CN.md` + `.gitignore` + `results/`，全部都已包含在 ZIP 里。

你已有的检查/训练/绘图/错误分析命令与 README 对应，不必再次训练才可以上传 GitHub。
