"""
Скрипт для подготовки собственных данных для суммаризации
Упрощенная версия для работы с небольшим количеством статей
"""

import sys
import os
import json
import pickle as pkl
from collections import Counter
from typing import List, Tuple, Optional
import re


class CustomDatasetPreparer:
    """
    Класс для подготовки собственного датасета для задачи суммаризации
    """

    def __init__(self, input_dir: str, output_dir: str = "prepared_data"):
        """
        Параметры:
        - input_dir: директория с вашими текстовыми файлами
        - output_dir: директория для сохранения подготовленных данных
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.finished_files_dir = os.path.join(output_dir, "finished_files")

        # Создаем необходимые директории
        os.makedirs(self.finished_files_dir, exist_ok=True)

        # Счетчик для словаря
        self.vocab_counter = Counter()

    def read_custom_file(self, filepath: str) -> Tuple[List[str], List[str]]:
        """
        Читает файл с вашим форматом данных

        Ожидаемый формат файла:
        1. Текст статьи (может быть несколько абзацев)
        2. Разделитель: строка с "===SUMMARY===" или "@highlight"
        3. Краткое изложение (может быть несколько предложений)

        Возвращает:
        - article_sentences: список предложений статьи
        - summary_sentences: список предложений суммаризации
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Попробуем разные варианты разделителей
        separators = ['===SUMMARY===', '===ABSTRACT===', '@highlight', '---']

        article_text = content
        summary_text = ""

        for separator in separators:
            if separator in content:
                parts = content.split(separator, 1)
                article_text = parts[0].strip()
                summary_text = parts[1].strip() if len(parts) > 1 else ""
                break

        # Если разделитель не найден, попробуем другой подход
        if not summary_text:
            # Можно предположить, что последний абзац - это суммаризация
            paragraphs = content.strip().split('\n\n')
            if len(paragraphs) > 1:
                article_text = '\n\n'.join(paragraphs[:-1])
                summary_text = paragraphs[-1]
                print(f"Внимание: в файле {filepath} не найден явный разделитель. "
                      f"Используем последний абзац как суммаризацию.")
            else:
                print(f"Внимание: в файле {filepath} не удалось выделить суммаризацию. "
                      f"Используем весь текст как статью.")
                article_text = content
                summary_text = "Суммаризация отсутствует."

        # Разбиваем на предложения
        article_sentences = self.split_into_sentences(article_text)
        summary_sentences = self.split_into_sentences(summary_text)

        return article_sentences, summary_sentences

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбивает текст на предложения

        Параметры:
        - text: исходный текст

        Возвращает:
        - список предложений
        """
        # Простое разбиение по точкам, восклицательным и вопросительным знакам
        # Можно заменить на более сложную логику или использовать NLTK

        # Нормализуем пробелы
        text = ' '.join(text.split())

        # Разбиваем по концам предложений
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Фильтруем пустые строки и нормализуем
        sentences = [s.strip() for s in sentences if s.strip()]

        # Добавляем точки, если их нет
        sentences = [s if s[-1] in '.!?' else s + '.' for s in sentences if s]

        return sentences

    def process_file(self, filepath: str, file_id: int) -> dict:
        """
        Обрабатывает один файл и возвращает словарь с данными

        Параметры:
        - filepath: путь к файлу
        - file_id: идентификатор файла

        Возвращает:
        - словарь с полями 'id', 'article', 'abstract'
        """
        article_sentences, summary_sentences = self.read_custom_file(filepath)

        # Приводим к нижнему регистру
        article_sentences = [s.lower() for s in article_sentences]
        summary_sentences = [s.lower() for s in summary_sentences]

        # Обновляем словарь
        all_words = ' '.join(article_sentences + summary_sentences).split()
        self.vocab_counter.update(all_words)

        return {
            'id': str(file_id),
            'article': article_sentences,
            'abstract': summary_sentences,
            'filename': os.path.basename(filepath)
        }

    def create_splits(self, files: List[str],
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
        """
        Разделяет файлы на train/val/test выборки

        Параметры:
        - files: список файлов
        - train_ratio: доля обучающей выборки
        - val_ratio: доля валидационной выборки

        Возвращает:
        - кортеж (train_files, val_files, test_files)
        """
        import random
        random.shuffle(files)

        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # Если файлов мало, убедимся, что хотя бы по одному в каждой выборке
        if n_files >= 3:
            if not val_files and n_files > n_train:
                val_files = [test_files.pop(0)] if test_files else []
            if not test_files and n_files > n_train + len(val_files):
                test_files = [val_files.pop()] if len(val_files) > 1 else []

        return train_files, val_files, test_files

    def save_split(self, files: List[str], split_name: str, start_id: int = 0):
        """
        Сохраняет файлы одной выборки в JSON формате

        Параметры:
        - files: список файлов для обработки
        - split_name: имя выборки ('train', 'val', 'test')
        - start_id: начальный ID для нумерации
        """
        split_dir = os.path.join(self.finished_files_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\nОбработка {split_name} выборки ({len(files)} файлов)...")

        for i, filepath in enumerate(files):
            file_id = start_id + i

            try:
                # Обрабатываем файл
                data = self.process_file(filepath, file_id)

                # Сохраняем в JSON
                output_path = os.path.join(split_dir, f"{i}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"  ✓ {os.path.basename(filepath)} -> {output_path}")

            except Exception as e:
                print(f"  ✗ Ошибка при обработке {filepath}: {e}")

    def save_vocabulary(self, min_count: int = 2):
        """
        Сохраняет словарь

        Параметры:
        - min_count: минимальная частота слова для включения в словарь
        """
        # Фильтруем редкие слова
        filtered_vocab = {word: count for word, count in self.vocab_counter.items()
                          if count >= min_count}

        # Сохраняем в pickle
        vocab_path = os.path.join(self.finished_files_dir, "vocab_cnt.pkl")
        with open(vocab_path, 'wb') as f:
            pkl.dump(Counter(filtered_vocab), f)

        # Также сохраняем в текстовом виде для удобства
        vocab_txt_path = os.path.join(self.finished_files_dir, "vocab.txt")
        with open(vocab_txt_path, 'w', encoding='utf-8') as f:
            for word, count in sorted(filtered_vocab.items(),
                                      key=lambda x: x[1], reverse=True):
                f.write(f"{word}\t{count}\n")

        print(f"\nСловарь сохранен:")
        print(f"  - Всего уникальных слов: {len(self.vocab_counter)}")
        print(f"  - Слов с частотой >= {min_count}: {len(filtered_vocab)}")
        print(f"  - Файлы: {vocab_path}, {vocab_txt_path}")

    def prepare_dataset(self, train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        file_extension: str = '.txt'):
        """
        Основной метод для подготовки всего датасета

        Параметры:
        - train_ratio: доля обучающей выборки
        - val_ratio: доля валидационной выборки
        - file_extension: расширение файлов для обработки
        """
        print(f"Подготовка датасета из директории: {self.input_dir}")
        print(f"Результаты будут сохранены в: {self.output_dir}")

        # Получаем список файлов
        files = []
        for filename in os.listdir(self.input_dir):
            if filename.endswith(file_extension):
                files.append(os.path.join(self.input_dir, filename))

        if not files:
            print(f"Не найдено файлов с расширением {file_extension} в {self.input_dir}")
            return

        print(f"Найдено файлов: {len(files)}")

        # Разделяем на выборки
        train_files, val_files, test_files = self.create_splits(
            files, train_ratio, val_ratio
        )

        print(f"\nРазделение на выборки:")
        print(f"  - Train: {len(train_files)} файлов")
        print(f"  - Val: {len(val_files)} файлов")
        print(f"  - Test: {len(test_files)} файлов")

        # Обрабатываем и сохраняем каждую выборку
        self.save_split(train_files, 'train', start_id=0)
        self.save_split(val_files, 'val', start_id=len(train_files))
        self.save_split(test_files, 'test', start_id=len(train_files) + len(val_files))

        # Сохраняем словарь
        self.save_vocabulary()

        print("\nПодготовка датасета завершена!")


def create_example_file(output_dir: str = "example_data"):
    """
    Создает примеры файлов для демонстрации формата
    """
    os.makedirs(output_dir, exist_ok=True)

    examples = [
        {
            'filename': 'article1.txt',
            'content': """Искусственный интеллект продолжает развиваться быстрыми темпами. 
Новые модели машинного обучения становятся все более мощными и эффективными. 
Компании по всему миру инвестируют миллиарды долларов в исследования ИИ.

===SUMMARY===
ИИ быстро развивается, привлекая огромные инвестиции от компаний по всему миру."""
        },
        {
            'filename': 'article2.txt',
            'content': """Изменение климата остается одной из главных проблем человечества.
Ученые предупреждают о необходимости срочных действий для сокращения выбросов.
Многие страны принимают новые экологические стандарты.

===SUMMARY===
Изменение климата требует срочных действий и новых экологических стандартов."""
        }
    ]

    for example in examples:
        filepath = os.path.join(output_dir, example['filename'])
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(example['content'])

    print(f"Примеры файлов созданы в директории: {output_dir}")
    print("Используйте этот формат для ваших данных.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nИСПОЛЬЗОВАНИЕ:")
        print("python prepare_custom_data.py <директория_с_вашими_файлами> [директория_для_результатов]")
        print("\nПример:")
        print("python prepare_custom_data.py my_articles prepared_data")
        print("\nДля создания примеров файлов:")
        print("python prepare_custom_data.py --create-examples")
        sys.exit(1)

    if sys.argv[1] == '--create-examples':
        create_example_file()
        sys.exit(0)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "prepared_data"

    if not os.path.exists(input_dir):
        print(f"Директория {input_dir} не существует!")
        sys.exit(1)

    # Создаем и запускаем препаратор
    preparer = CustomDatasetPreparer(input_dir, output_dir)
    preparer.prepare_dataset()