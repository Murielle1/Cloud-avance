# ── Imports ──────────────────────────────────────────
import boto3
import pandas as pd
import io

# ── Config ────────────────────────────────────────────
BUCKET     = "rotten-tomatoes-project"
RAW        = "raw/"
OUT        = "processed/"
CHUNK_SIZE = 50_000  # lignes lues à la fois

s3 = boto3.client("s3")

def read_csv_full(key, cols):
    """Lit un CSV complet depuis S3 (pour les petits fichiers)."""
    obj = s3.get_object(Bucket=BUCKET, Key=RAW + key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), usecols=cols)

def read_csv_chunked(key, cols):
    """Lit un gros CSV depuis S3 par morceaux pour économiser la RAM."""
    obj = s3.get_object(Bucket=BUCKET, Key=RAW + key)
    raw_bytes = obj["Body"].read()
    chunks = []
    for chunk in pd.read_csv(
        io.BytesIO(raw_bytes),
        usecols=cols,
        chunksize=CHUNK_SIZE,
        low_memory=True
    ):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# ── 1. Chargement ─────────────────────────────────────
print("Chargement des films (petit fichier)...")
movies_cols = ["rotten_tomatoes_link", "genres", "runtime"]
df_movies = read_csv_full("rotten_tomatoes_movies.csv", movies_cols)
print(f"  → {len(df_movies)} films chargés")

print("Chargement des critiques par chunks (gros fichier)...")
critics_cols = [
    "rotten_tomatoes_link",
    "review_content",   # texte de l'avis  → feature principale
    "review_score",     # note du critique  → feature signal
    "review_type",      # Fresh / Rotten    → CIBLE (y)
]
df_critics = read_csv_chunked("rotten_tomatoes_critic_reviews.csv", critics_cols)
print(f"  → {len(df_critics)} avis chargés")

# ── 2. Jointure ───────────────────────────────────────
print("Jointure des deux fichiers...")
df = df_critics.merge(df_movies, on="rotten_tomatoes_link", how="inner")
print(f"  → {len(df)} lignes après jointure")

del df_critics, df_movies  # libère la RAM

# ── 3. Nettoyage de la cible review_type ─────────────
print("Nettoyage de review_type...")

# Normalise les valeurs : strip + lowercase pour éviter les variantes
df["review_type"] = df["review_type"].str.strip().str.capitalize()

# Garde uniquement Fresh et Rotten (supprime les NaN et valeurs inconnues)
df = df[df["review_type"].isin(["Fresh", "Rotten"])]

# Encodage numérique : Fresh = 1, Rotten = 0  (requis par SageMaker)
df["label"] = df["review_type"].map({"Fresh": 1, "Rotten": 0})

# ── 4. Nettoyage du texte des avis ───────────────────
print("Nettoyage du texte des avis...")
df = df.dropna(subset=["review_content"])

df["review_content"] = (
    df["review_content"]
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", "", regex=True)
    .str.strip()
)

# Supprime les avis trop courts (bruit pour le NLP)
df = df[df["review_content"].str.len() > 20]

# ── 5. Nettoyage des features secondaires ────────────
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")

# ── 6. Colonnes finales ───────────────────────────────
df_final = df[[
    "review_content",  # feature principale (texte)
    "review_score",    # feature signal (note critique)
    "genres",          # feature catégorielle
    "runtime",         # feature numérique
    "review_type",     # cible lisible (Fresh / Rotten)
    "label",           # cible encodée (1 / 0) pour SageMaker
]].copy()

del df  # libère la RAM

print(f"  → {len(df_final)} lignes propres")
print(f"  → Répartition : {df_final['review_type'].value_counts().to_dict()}")

# ── 7. Sauvegarde dans S3 processed/ ─────────────────
print("Sauvegarde dans S3...")
buf = io.StringIO()
df_final.to_csv(buf, index=False)
s3.put_object(
    Bucket=BUCKET,
    Key=OUT + "clean_reviews.csv",
    Body=buf.getvalue().encode()
)

print(f"ETL terminé — {len(df_final)} lignes dans s3://{BUCKET}/{OUT}clean_reviews.csv")