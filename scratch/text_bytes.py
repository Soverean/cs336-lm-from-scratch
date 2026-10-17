text = "猫cat"

print("原始文本：", text)
print("字符串类型：", type(text))
print("字符串长度：", len(text))

data = text.encode("utf-8")

print("编码后的字节：", data)
print("字节类型：", type(data))
print("字节长度：", len(data))
print("每个字节的整数值：", list(data))

restored = data.decode("utf-8")
print("解码后的文本：", restored)
print("是否还原成功：", restored == text)

print(ord('牛'))
print(chr(0))
print('this is a test'+chr(0)+'string')