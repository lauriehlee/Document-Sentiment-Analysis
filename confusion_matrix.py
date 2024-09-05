

def confusion_matrix(tup):
    tp, fn, fp, tn = tup
    tp_string = "TP: " + str(tp)
    fn_string = "FN: " + str(fn)
    fp_string = "FP: " + str(fp)
    tn_string = "TN: " + str(tn)
    labels = [" ", "  +  ", "  -  ", "+", tp_string, fn_string, "-", fp_string, tn_string]


    table = [[None, None, None],
             [None, None, None],
             [None, None, None]]


    for i in range(3):
        for j in range(3):
            table[i][j] = labels[i*3 + j]

    # Print the table
    for row in table:
        print("|", end="")
        for item in row:
            print(f" {item} |", end="")
        print()


# Example usage
labels = ["A", "B", "C",
          "D", "E", "F",
          "G", "H", "I"]

confusion_matrix((632, 11868, 34, 12466))