import argparse
import logging
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from keybert import KeyBERT
from scipy.special import softmax
from sentence_transformers import SentenceTransformer, CrossEncoder

# Конфигурация логирования
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TextProcessor:
    """Класс для предобработки и разбиения текста на куски."""

    @staticmethod
    def clean_text(raw: str) -> str:
        """Удаляет переносы и лишние пробелы."""
        text = re.sub(r'-\s*\n\s*', '', raw)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
        """Разбивает текст на куски примерно по chunk_size слов."""
        words = text.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]


class KeywordDocumentRanker:
    """Класс, отвечающий за извлечение ключевых слов, вычисление эмбеддингов и ранжирование."""

    def __init__(self, pdf_files: list[dict], cache_dir: Path):
        self.pdf_files = pdf_files
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.keywords_path = cache_dir / "keywords.pkl"
        self.embeddings_path = cache_dir / "docs_emb.pkl"
        self.df_path = cache_dir / "keyword_doc_df.pkl"

        self.titles = [p['title'] for p in pdf_files]
        self.docs_chunks: list[list[str]] = []
        self.docs_texts: list[str] = []
        self.keywords: list[str] = []
        self.docs_emb: list = []
        self.df: pd.DataFrame

    def load_or_compute(self):
        # Сначала читаем и обрабатываем документы, чтобы docs_texts всегда были доступны
        self._read_and_process()

        if self.keywords_path.exists() and self.embeddings_path.exists() and self.df_path.exists():
            self._load_cache()
        else:
            self._extract_keywords()
            self._compute_embeddings_and_rank()
            self._save_cache()

    def _load_cache(self):
        logger.info("Загружаем кэшированные данные...")
        with open(self.keywords_path, 'rb') as f:
            self.keywords = pickle.load(f)
        with open(self.embeddings_path, 'rb') as f:
            self.docs_emb = pickle.load(f)
        self.df = pd.read_pickle(self.df_path)
        logger.info("Кэш загружен.")

    def _read_and_process(self):
        logger.info("Читаем и обрабатываем документы...")
        self.docs_chunks.clear()
        self.docs_texts.clear()
        for paper in self.pdf_files:
            raw = Path(paper['file']).read_text(encoding='utf-8')
            clean = TextProcessor.clean_text(raw)
            chunks = TextProcessor.chunk_text(clean, chunk_size=500)
            self.docs_chunks.append(chunks)
            self.docs_texts.append(" ".join(chunks))
        logger.info("Документы обработаны.")

    def _extract_keywords(self):
        logger.info("Извлекаем ключевые фразы...")
        kw_model = KeyBERT('all-mpnet-base-v2')
        all_phrases = []
        for text in self.docs_texts:
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=5
            )
            all_phrases.extend([phrase for phrase, _ in kws])

        freq = Counter(all_phrases)
        self.keywords = [kw for kw, _ in freq.most_common(10)]
        logger.info("Ключевые фразы: %s", self.keywords)

    def _compute_embeddings_and_rank(self):
        logger.info("Вычисляем эмбеддинги документов...")
        encoder = SentenceTransformer('msmarco-distilbert-base-tas-b')
        self.docs_emb = [
            encoder.encode(chunks, convert_to_numpy=True).mean(axis=0)
            for chunks in self.docs_chunks
        ]

        logger.info("Ранжируем документы с помощью Cross-Encoder...")
        cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [
            (kw, doc_text)
            for kw in self.keywords
            for doc_text in self.docs_texts
        ]
        scores = cross.predict(pairs).reshape(
            len(self.keywords), len(self.docs_texts)
        )
        probs = softmax(scores, axis=1)
        self.df = pd.DataFrame(probs, index=self.keywords, columns=self.titles)

    def _save_cache(self):
        logger.info("Сохраняем кэш...")
        with open(self.keywords_path, 'wb') as f:
            pickle.dump(self.keywords, f)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.docs_emb, f)
        self.df.to_pickle(self.df_path)
        logger.info("Кэш сохранён.")

    def get_ranking(self, keyword: str) -> pd.Series:
        """Возвращает series с вероятностями документов по ключевому слову."""
        if keyword in self.df.index:
            return self.df.loc[keyword]
        # для новых ключевых слов используем Cross-Encoder по текстам
        cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = [
            cross.predict([(keyword, doc)])[0]
            for doc in self.docs_texts
        ]
        probs = softmax([scores], axis=1)[0]
        return pd.Series(probs, index=self.titles)


def main():
    parser = argparse.ArgumentParser(
        description="Ранжирование документов по ключевым словам"
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('cache'),
        help='Папка для хранения кэша'
    )
    args = parser.parse_args()

    pdfs = [
        {'title': "Attention Is All You Need", 'file': "C:/Users/PC/Downloads/for_student/test/Attention Is All You Need.txt"},
        {'title': "Deep Residual Learning", 'file': "C:/Users/PC/Downloads/for_student/test/Deep Residual Learning.txt"},
        {'title': "BERT", 'file': "C:/Users/PC/Downloads/for_student/test/BERT.txt"},
        {'title': "GPT-3", 'file': "C:/Users/PC/Downloads/for_student/test/GPT-3.txt"},
        {'title': "Adam Optimizer", 'file': "C:/Users/PC/Downloads/for_student/test/Adam Optimizer.txt"},
        {'title': "GANs", 'file': "C:/Users/PC/Downloads/for_student/test/GANs.txt"},
        {'title': "U-Net", 'file': "C:/Users/PC/Downloads/for_student/test/U-Net.txt"},
        {'title': "DALL-E 2", 'file': "C:/Users/PC/Downloads/for_student/test/DALL-E 2.txt"},
        {'title': "Stable Diffusion", 'file': "C:/Users/PC/Downloads/for_student/test/Stable Diffusion.txt"},
        {'title': "check", 'file': "C:/Users/PC/Downloads/for_student/test/check.txt"},
    ]

    ranker = KeywordDocumentRanker(pdfs, args.cache_dir)
    ranker.load_or_compute()

    logger.info("Введите ключевые слова через запятую:")
    try:
        while True:
            user_input = input("> ")
            kws = [kw.strip() for kw in user_input.split(',') if kw.strip()]
            scores = Counter()
            for kw in kws:
                ranking = ranker.get_ranking(kw)
                adjusted = ranking.apply(lambda v: v if v >= 0.35 else 0)
                for title, val in adjusted.items():
                    scores[title] += val

            selected = [t for t, s in scores.items() if s >= 0.8]
            if selected:
                print("Подходящие документы:")
                for title in selected:
                    print(f" - {title}")
            else:
                print("Документы не найдены.")
    except KeyboardInterrupt:
        print("\nВыход.")


if __name__ == '__main__':
    main()
