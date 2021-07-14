from sudoku_solver.models import OCRNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, help="path to place trained model")

args = vars(parser.parse_args())

START_LEARN_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 128

logger.info("Loading MNIST dataset")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.fit_transform(testLabels)

logger.info("Building model")
opt = Adam(learning_rate=START_LEARN_RATE)
model = OCRNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

logger.info("Training model")
H = model.fit(
    trainData,
    trainLabels,
    validation_data=(testData, testLabels),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
)

logger.info("Testing model")
predictions = model.predict(testData)
logger.info(
    classification_report(
        testLabels.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in le.classes_],
    )
)

logger.info(f"Serializing and saving model to: {args['output']}")
model.save(args["output"], save_format="h5")
