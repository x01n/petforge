#!/usr/bin/env bash
# 清理 Git 对大型二进制运行时资源的跟踪。
#
# 用途：仓库已把 GPT-Sovits / models / sprites / dic / vits_models 等
#       大型资源目录列进 .gitignore，但历史提交中若已跟踪会继续占仓。
#       运行本脚本后这些路径将变为“删除 + 忽略”，不会再随提交入库。
#
# 适用：准备把仓库推送到远程（GitHub 等有单文件 100M 限制的托管）之前。
# 注意：脚本会从索引中移除这些文件（保留工作区文件本体），
#       风险仅影响 git 历史；请确认你想保留哪些"误追"再执行。
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 当前 .gitignore 已覆盖的大型资源路径 =="
for p in GPT-Sovits models sprites dic vits_models; do
  tracked=$(git ls-files "$p" | wc -l)
  echo "  $p: 跟踪 $tracked 个文件"
done

echo
echo "== 从索引移除（-r --cached 保留工作区文件）=="
for p in GPT-Sovits models sprites dic vits_models; do
  if [ "$(git ls-files "$p" | wc -l)" -gt 0 ]; then
    git rm -r --cached --quiet "$p"
    echo "  已移除索引: $p"
  else
    echo "  跳过（未跟踪）: $p"
  fi
done

echo
echo "== 结果 =="
git status --short | head -30
echo "（上方红色项即被移除跟踪的资源，业务代码保持不变。）"