from sklearn.base import BaseEstimator, TransformerMixin

class DisciplineExtractor(BaseEstimator, TransformerMixin):
    """Extract the discipline from data"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        transformed = []
        for d in data:
            transformed.append(d['discipline'])
        return transformed


class GeneralExtractor(BaseEstimator, TransformerMixin):
    """Extract and merge all fields from a sample to a string"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        transformed = []
        for d in data:
            transformed.append(' '.join(d.values()))
        return transformed