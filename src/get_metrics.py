import sys
import pandas as pd
import metrics

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USE: python %s train_csv_pred_file test_csv_pred_file" % sys.argv[0])
        sys.exit(0)

    train_df = pd.read_csv(sys.argv[1])
    test_df = pd.read_csv(sys.argv[2])

    tra_acc = metrics.accuracy(train_df["target"], train_df["prediction"])
    tra_tpr_mean = metrics.tpr_mean(train_df["target"], train_df["prediction"])
    tra_gm = metrics.gm(train_df["target"], train_df["prediction"])

    tst_acc = metrics.accuracy(test_df["target"], test_df["prediction"])
    tst_tpr_mean = metrics.tpr_mean(test_df["target"], test_df["prediction"])
    tst_gm = metrics.gm(test_df["target"], test_df["prediction"])

    print("TRAIN ACC: %.4f" % tra_acc)
    print("TRAIN TPR MEAN: %.4f" % tra_tpr_mean)
    print("TRAIN GM: %.4f" % tra_gm)

    print("TEST ACC: %.4f" % tst_acc)
    print("TEST TPR MEAN: %.4f" % tst_tpr_mean)
    print("TEST GM: %.4f" % tst_gm)
