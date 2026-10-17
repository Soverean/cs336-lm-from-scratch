from pathlib import Path
from cs336_basics import config


def make_subset(input_path, output_path,max_bytes):

    with input_path.open("rb") as f:
        data = f.read(max_bytes)

    print(type(data))
    print(len(data))

    eos = b"<|endoftext|>"
    boundary = data.rfind(eos)

    if boundary == -1:
        raise ValueError('没有找到完整的文档结束符')

    end_position = boundary + len(eos)
    subset = data[0:end_position]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(subset)

    print("保存位置：", output_path)
    print("实际字节数：", len(subset))
    print("实际 MiB：", len(subset) / (1024 * 1024))
    print("结尾是否完整：", subset.endswith(eos))

if __name__ == '__main__':

    data_dir = Path(
        "/Users/soverean/cs336-a1-2025/data/tinystories"
    )

    train_path = data_dir / "TinyStoriesV2-GPT4-train.txt"
    sample_path = data_dir / "train_5MiB.txt"


    make_subset(
        train_path,
        data_dir / "train_5MiB.txt",
        5 * 1024 * 1024,
    )




