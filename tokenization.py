#train_split = 20000

#train_val_files = [(x, 0) for x in glob.glob("aclImdb/train/neg/*.txt")] + [(x, 1) for x in glob.glob("aclImdb/train/pos/*.txt")]
#shuffle(train_val_files)
#test_files = [(x, 0) for x in glob.glob("aclImdb/test/neg/*.txt")] + [(x, 1) for x in glob.glob("aclImdb/test/pos/*.txt")]

#train_files  = train_val_files[:train_split].copy()
#val_files = train_val_files[train_split:].copy()

#train_targets = [x[1] for x in train_files]
#val_targets = [x[1] for x in val_files]
#test_targets = [x[1] for x in test_files]

#train_data = [open(filename[0]).read().replace("<br />", "").replace("\'", "") for filename in train_files]
#val_data = [open(filename[0]).read().replace("<br />", "").replace("\'", "") for filename in val_files]
#test_data = [open(filename[0]).read().replace("<br />", "").replace("\'", "") for filename in test_files]


#pkl.dump(train_targets, open("train_targets.p", "wb"))
#pkl.dump(val_targets, open("val_targets.p", "wb"))
#pkl.dump(test_targets, open("test_targets.p", "wb"))

#ORIGINAL TOKENIZATION
# val set tokens
#print ("Tokenizing val data")
#val_data_tokens, _ = tokenize_dataset(val_data, False)
#pkl.dump(val_data_tokens, open("val_data_tokens.p", "wb"))

# test set tokens
#print ("Tokenizing test data")
#test_data_tokens, _ = tokenize_dataset(test_data, False)
#pkl.dump(test_data_tokens, open("test_data_tokens.p", "wb"))

# train set tokens
#print ("Tokenizing train data")
#train_data_tokens, all_train_tokens = tokenize_dataset(train_data, True)
#pkl.dump(train_data_tokens, open("train_data_tokens.p", "wb"))
#pkl.dump(all_train_tokens, open("all_train_tokens.p", "wb"))

#NO STOP WORDS
# val set tokens
#print ("Tokenizing val data")
#val_data_tokens_no_stop, _ = tokenize_dataset(val_data, False, True, False)
#pkl.dump(val_data_tokens_no_stop, open("val_data_tokens_no_stop.p", "wb"))

# test set tokens
#print ("Tokenizing test data")
#test_data_tokens_no_stop, _ = tokenize_dataset(test_data, False, True, False)
#pkl.dump(test_data_tokens_no_stop, open("test_data_tokens_no_stop.p", "wb"))

# train set tokens
#print ("Tokenizing train data")
#train_data_tokens_no_stop, all_train_tokens_no_stop = tokenize_dataset(train_data, True, True, False)
#pkl.dump(train_data_tokens_no_stop, open("train_data_tokens_no_stop.p", "wb"))
#pkl.dump(all_train_tokens_no_stop, open("all_train_tokens_no_stop.p", "wb"))

#ENTITY TOKENIZATION
# val set tokens
#print ("Tokenizing val data")
#val_data_tokens_entity_keep, _ = tokenize_dataset(val_data, False, False, True)
#pkl.dump(val_data_tokens_entity_keep, open("val_data_tokens_entity_keep.p", "wb"))

# test set tokens
#print ("Tokenizing test data")
#test_data_tokens_entity_keep, _ = tokenize_dataset(test_data, False, False, True)
#pkl.dump(test_data_tokens_entity_keep, open("test_data_tokens_entity_keep.p", "wb"))

# train set tokens
#print ("Tokenizing train data")
#train_data_tokens_entity_keep, all_train_tokens_entity_keep = tokenize_dataset(train_data, True, False, True)
#pkl.dump(train_data_tokens_entity_keep, open("train_data_tokens_entity_keep.p", "wb"))
#pkl.dump(all_train_tokens_entity_keep, open("all_train_tokens_entity_keep.p", "wb"))