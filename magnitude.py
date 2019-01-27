import pymagnitude


class Word_vectors:
    def __init__(self, fname):
        self._vectors = pymagnitude.Magnitude(fname, lazy_loading=-1, blocking=True)

    def get_similarities(self, key_word, word_list):
        return self._vectors.similarity(key_word, word_list)

    def get_nearest_neighbors(self, key_word, num_results):
        return self._vectors.most_similar(key_word, topn = num_results)

if __name__ == "__main__":
    vectors = Word_vectors("/Users/kelliedinh/Desktop/scalar-adjectives/data/glove.6B.50d.magnitude")
    results = vectors.get_nearest_neighbors("cat", 10)
    print(results)