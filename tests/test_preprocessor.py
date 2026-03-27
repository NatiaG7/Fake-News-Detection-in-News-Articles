import unittest
from text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_lowercase(self):
        self.assertEqual(self.preprocessor.lowercase('Hello World'), 'hello world')

    def test_url_removal(self):
        text = 'Visit https://example.com for more info.'
        expected = 'Visit  for more info.'
        self.assertEqual(self.preprocessor.remove_urls(text), expected)

    def test_punctuation_removal(self):
        text = 'Hello, World!'
        expected = 'Hello World'
        self.assertEqual(self.preprocessor.remove_punctuation(text), expected)

    def test_stopword_removal(self):
        text = 'This is a simple test.'
        expected = 'simple test.'
        self.assertEqual(self.preprocessor.remove_stopwords(text), expected)

    def test_lemmatization(self):
        text = 'running runs ran'
        expected = 'run run run'
        self.assertEqual(self.preprocessor.lemmatize(text), expected)

    def test_full_pipeline(self):
        text = 'Visit https://example.com for running and runs.'
        expected = 'visit for run and run.'
        processed = self.preprocessor.full_pipeline(text)
        self.assertEqual(processed, expected)

if __name__ == '__main__':
    unittest.main()