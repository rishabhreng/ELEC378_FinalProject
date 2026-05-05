# import a bunch of stuff
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# import a bunch of stuff from our common module
from classify_common import (
    FILE_PATH,
    FeatureExtractor,
    RUNS_DIR,
    SEED,
    build_classical_feature_matrix,
    load_test_image,
    load_metadata,
    get_submission_image_ids,
    set_seed,
    split_dataset,
)

# Function to train a HOG + linear SVM classifier and save the model path for prediction
def train_classical_classifier(run_name: str) -> Path:
    df = load_metadata() # load the metadata into a pandas df
    dataset = split_dataset(df, val_size=0.2, random_state=SEED) # split the df into train and validation sets

    # create directory for the run and define model path
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "hog_linear_svm.joblib" # joblib file to save the model for prediction

    # build feature matrices for HOG + color histogram features
    extractor = FeatureExtractor()
    # training and validation matrices
    x_train = build_classical_feature_matrix(dataset.train_df, extractor)
    x_val = build_classical_feature_matrix(dataset.val_df, extractor)
    y_train = dataset.train_df["TARGET"].map(dataset.class_to_index).to_numpy()
    y_val = dataset.val_df["TARGET"].map(dataset.class_to_index).to_numpy()

    # train a linear SVM classifier on the HOG + color histogram features
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(C=2.0, class_weight="balanced", max_iter=12000),
    )
    # test the classifier on the validation set to print accuracy score
    classifier.fit(x_train, y_train)
    val_predictions = classifier.predict(x_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"[HOG-SVM] Model accuracy: {val_accuracy:.4f}")

    # save the model
    joblib.dump(
        {
            "classifier": classifier,
            "class_names": dataset.class_names,
            "feature_size": extractor.feature_size,
        },
        model_path,
    )
    return model_path

# Function to predict labels for the test set using our classifier and save the submission file
def predict_classical_labels(model_path: Path, output_path: Path) -> pd.DataFrame:
    # load the trained model
    checkpoint = joblib.load(model_path)
    classifier = checkpoint["classifier"]
    class_names = list(checkpoint["class_names"])
    extractor = FeatureExtractor(feature_size=int(checkpoint["feature_size"]))

    # build feature matrix for test set and predict the labels
    image_ids = get_submission_image_ids()
    features = [
        extractor.transform(load_test_image(image_id))
        for image_id in image_ids
    ]
    x_test = np.stack(features, axis=0)
    predicted_indices = classifier.predict(x_test)
    predictions = [class_names[int(index)] for index in predicted_indices] # map predicted indices back to class names

    # write the submission csv file
    submission = pd.DataFrame({"ID": image_ids, "TARGET": predictions})
    submission.to_csv(output_path, index=False)
    print(f"Wrote submission file to {output_path}")
    return submission


def main():
    set_seed(SEED) # set the random seed (42) for reproducibility
    model_path = train_classical_classifier(run_name="hog_linear_svm_butterflies") # train hog+svm classifier
    predict_classical_labels(model_path=model_path, output_path=FILE_PATH / "submission_hog_svm.csv") # predict labels and write submission file


if __name__ == "__main__":
    main()
