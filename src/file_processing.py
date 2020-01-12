import os


# File relative path
file_path_train = "../../aclImdb/train/"
file_path_test = "../../aclImdb/test/"

# Remove the old files if existed in memory
def remove_old_files_if_existed():
    
    if os.path.exists(file_path_train + "train_reviews.txt"):
        os.remove(file_path_train + "train_reviews.txt")
        print("Successfully removed [train_reviews.txt]")
    else:
        print("Can't find [train_reviews.txt]")
    
    if os.path.exists(file_path_train + "train_labels.txt"):
        os.remove(file_path_train + "train_labels.txt")
        print("Successfully removed [train_labels.txt]")
    else:
        print("Can't find [train_labels.txt]")
        
    if os.path.exists(file_path_test + "test_reviews.txt"):
        os.remove(file_path_test + "test_reviews.txt")
        print("Successfully removed [test_reviews.txt]")
    else:
        print("Can't find [test_reviews.txt]")
    
    if os.path.exists(file_path_test + "test_labels.txt"):
        os.remove(file_path_test + "test_labels.txt")
        print("Successfully removed [test_labels.txt]")
    else:
        print("Can't find [test_labels.txt]")

# Generate new input files
def generate_new_files():

    # Generate train input files "train_reviews.txt" and "train_labels.txt"
    for filename in os.listdir(file_path_train + "neg/"):
        with open(file_path_train + "neg/" + filename) as f:
            for line in f.readlines():
                with open(file_path_train + "train_reviews.txt","a") as train_reviews:
                    train_reviews.write(line + "\n")
                with open(file_path_train + "train_labels.txt","a") as train_labels:
                    train_labels.write("negative\n")
                    
    for filename in os.listdir(file_path_train + "pos/"):
        with open(file_path_train + "pos/" + filename) as f:
            for line in f.readlines():
                with open(file_path_train + "train_reviews.txt","a") as train_reviews:
                    train_reviews.write(line + "\n")
                with open(file_path_train + "train_labels.txt","a") as train_labels:
                    train_labels.write("positive\n")        
                    
    print("Successfully generated [train_reviews.txt] and [train_labels.txt]")     
    
    # Generate test input files "test_reviews.txt" and "test_labels.txt"
    for filename in os.listdir(file_path_test + "neg/"):
        with open(file_path_test + "neg/" + filename) as f:
            for line in f.readlines():
                with open(file_path_test + "test_reviews.txt","a") as test_reviews:
                    test_reviews.write(line + "\n")
                with open(file_path_test + "test_labels.txt","a") as test_labels:
                    test_labels.write("negative\n")
                    
    for filename in os.listdir(file_path_test + "pos/"):
        with open(file_path_test + "pos/" + filename) as f:
            for line in f.readlines():
                with open(file_path_test + "test_reviews.txt","a") as test_reviews:
                    test_reviews.write(line + "\n")
                with open(file_path_test + "test_labels.txt","a") as test_labels:
                    test_labels.write("positive\n")
                    
    print("Successfully generated [test_reviews.txt] and [test_labels.txt]")       
                
                