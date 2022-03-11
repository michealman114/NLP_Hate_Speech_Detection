import wget


data = wget.download('https://raw.githubusercontent.com/sjtuprog/fox-news-comments/master/fox-news-comments.json')

modern_data = wget.download('https://raw.githubusercontent.com/michealman114/NLP_Hate_Speech_Detection/main/modern_comments.json')

print(data)
print(modern_data)