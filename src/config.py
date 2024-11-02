# Konfigurasi variabel penting
RANDOM_STATE = 42
TEST_SIZE = 0.2
PARAM_GRID = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
