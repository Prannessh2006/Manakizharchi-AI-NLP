warnings.filterwarnings('ignore')
sns.set(style="whitegrid")


DATA_PATH = "data/reviews.csv"
# assert os.path.exists(DATA_PATH)
try:
    os.path.exists(DATA_PATH)
except AssertionError as e:
    print(f"Assertion error occured: {e}")
    

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
df = df.dropna(subset=['reviewText','overall'])
df['overall'] = df['overall'].astype(int)
df = df.reset_index(drop=True)

print(df['overall'].value_counts().sort_index())
print(df.head(3))


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


df['clean'] = df['reviewText'].apply(clean_text)
df['tokens'] = df['clean'].apply(tokenize_and_lemmatize)
df['clean_joined'] = df['tokens'].apply(lambda toks: " ".join(toks))


df['review_len'] = df['clean_joined'].apply(lambda x: len(x.split()))
print("Avg tokens:", df['review_len'].mean())

df = df[df['review_len'] >= 3].reset_index(drop=True)

plt.figure(figsize=(8,4))
sns.countplot(x='overall', data=df, order=sorted(df['overall'].unique()))
plt.title("Rating Distribution")
plt.show()

all_text = " ".join(df['clean_joined'].tolist())
wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(12,6)); plt.imshow(wc); plt.axis('off'); plt.show()

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df['clean_joined'])
y_reg = df['overall'].values

df['sentiment'] = (df['overall'] >= 4).astype(int)
y_clf = df['sentiment'].values


pca_n = 50

pca = PCA(n_components=min(pca_n, X_tfidf.shape[0])) 
X_pca = pca.fit_transform(X_tfidf.toarray())
print("PCA explained variance ratio (sum):", pca.explained_variance_ratio_.sum())

X = X_pca
