import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

try:
    from rich.console import Console
    from rich.tree import Tree
except ImportError:
    Console = None
    Tree = None

Stats = Dict[str, int]


def count_file(path: Path) -> Stats:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return {"files": 1, "lines": 0, "chars": 0}
    return {"files": 1, "lines": text.count("\n") + (0 if text.endswith("\n") or len(text) == 0 else 1), "chars": len(text)}


def merge_stats(a: Stats, b: Stats) -> Stats:
    return {"files": a["files"] + b["files"], "lines": a["lines"] + b["lines"], "chars": a["chars"] + b["chars"]}


def is_code_file(path: Path, extensions: Iterable[str], names: Iterable[str]) -> bool:
    if path.name in names:
        return True
    return path.suffix.lower() in extensions


def scan_dir(path: Path, extensions: Iterable[str], names: Iterable[str], exclude: Iterable[str]) -> Tuple[Stats, Dict[Path, Stats]]:
    stats: Stats = {"files": 0, "lines": 0, "chars": 0}
    subtree: Dict[Path, Stats] = {}

    for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.name in exclude:
            continue
        if child.is_dir():
            child_stats, child_subtree = scan_dir(child, extensions, names, exclude)
            if child_stats["files"] > 0:
                stats = merge_stats(stats, child_stats)
                subtree[child] = child_stats
                subtree.update(child_subtree)
        elif child.is_file() and is_code_file(child, extensions, names):
            file_stats = count_file(child)
            stats = merge_stats(stats, file_stats)
            subtree[child] = file_stats

    return stats, subtree


def format_stats(path: Path, stats: Stats) -> str:
    return f"{path.name} [files={stats['files']}, lines={stats['lines']}, chars={stats['chars']}]"


def print_ascii_tree(root: Path, subtree: Dict[Path, Stats], prefix: str = "") -> None:
    children = [p for p in sorted(subtree) if p.parent == root]
    for index, child in enumerate(children):
        connector = "└── " if index == len(children) - 1 else "├── "
        line_prefix = prefix + connector
        print(f"{line_prefix}{format_stats(child, subtree[child])}")
        if child.is_dir():
            child_children = [p for p in sorted(subtree) if p.parent == child]
            if child_children:
                extension = "    " if index == len(children) - 1 else "│   "
                print_ascii_tree(child, subtree, prefix + extension)


def print_rich_tree(root: Path, subtree: Dict[Path, Stats]) -> None:
    console = Console()

    def build_tree(node: Path, tree_node: Tree) -> None:
        children = [p for p in sorted(subtree) if p.parent == node]
        for child in children:
            label = f"[bold]{child.name}[/] [dim](files={subtree[child]['files']}, lines={subtree[child]['lines']}, chars={subtree[child]['chars']})[/]"
            branch = tree_node.add(label)
            if child.is_dir():
                build_tree(child, branch)

    root_label = f"[bold]{root.name}[/] [dim](files={subtree[root]['files'] if root in subtree else 0}, lines={subtree[root]['lines'] if root in subtree else 0}, chars={subtree[root]['chars'] if root in subtree else 0})[/]"
    rich_tree = Tree(root_label, guide_style="bold bright_blue")
    build_tree(root, rich_tree)
    console.print(rich_tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计后端代码目录的文件数、行数与字符数，并以树状格式显示。")
    parser.add_argument("root", nargs="?", default="..", help="要统计的根目录，默认 backend/. 相对于脚本位置。")
    parser.add_argument("--ext", nargs="*", default=[".py", ".yaml", ".yml", ".sh", ".env", ".ini", ".toml"],
                        help="要包含的文件扩展名列表，默认包含常见后端代码和配置文件。")
    parser.add_argument("--name", nargs="*", default=["Dockerfile", "Makefile"],
                        help="要包含的无扩展名文件名。")
    parser.add_argument("--exclude", nargs="*", default=["__pycache__", ".git", ".venv", "node_modules", "dist", "build", "target", "output", ".idea", ".vscode", ".cache", ".pytest_cache"],
                        help="要排除的目录名。")
    parser.add_argument("--pretty", action="store_true", help="使用 rich 输出漂亮的树状效果（需安装 rich）。")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.joinpath(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"目录不存在：{root}")

    stats, subtree = scan_dir(root, set(args.ext), set(args.name), set(args.exclude))
    print(format_stats(root, stats))
    if args.pretty and Console is not None and Tree is not None:
        subtree_with_root = dict(subtree)
        subtree_with_root[root] = stats
        print_rich_tree(root, subtree_with_root)
    elif args.pretty:
        print("warning: rich 未安装，已回退到纯文本输出。")
        print_ascii_tree(root, subtree)
    else:
        print_ascii_tree(root, subtree)


if __name__ == "__main__":
    main()
