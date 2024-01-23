import MeCab

mecab_non = MeCab.Tagger()

m_data = mecab_non.parse("pythonが大好きです。")
print(m_data)
