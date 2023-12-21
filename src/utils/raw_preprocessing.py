import pandas as pd
import os
import shutil
from langdetect import detect
from googletrans import Translator

def rename_images(image_folder):
    """
    Rename image files in the specified folder by keeping only the first two parts of the filename.

    Args:
    - image_folder (str): Path to the folder containing image files.

    Returns:
    - None
    """
    # List all files in the folder
    all_files = os.listdir(image_folder)

    for filename in all_files:
        # Split the filename into parts using '_' as a separator
        parts = filename.split('_')

        # Check if the filename has at least three parts (id1, id2, hash)
        if len(parts) >= 3:
            # Construct the new filename using the first two parts (id1_id2)


            new_filename = f"{parts[0]}_{parts[1]}.jpg"

            # Full path to the old and new files
            old_path = os.path.join(image_folder, filename)
            new_path = os.path.join(image_folder, new_filename)

            # Rename the file
            shutil.move(old_path, new_path)

def create_identifier(row):
    """
    Create an identifier by concatenating 'label' and 'cc' columns from a DataFrame row.

    Args:
    - row (pd.Series): A row from a DataFrame.

    Returns:
    - str: The concatenated identifier.
    """
    text = f"{int(row['label'])}_{row['cc']}"


    return text

def detect_language(sentence):
    """
    Detect the language of a given sentence.

    Args:
    - sentence (str): The input sentence.

    Returns:
    - str or None: The detected language or None if the language detection fails.
    """
    try:
        language = detect(sentence)
        return language
    except Exception as e:
        return None

def translate_to_english(sentence):
    """
    Translate a sentence to English.

    Args:
    - sentence (str): The input sentence.

    Returns:
    - str: The translated sentence in English.
    """
    translator = Translator()
    translation = translator.translate(sentence, dest='en')
    return translation.text

def translate_if_not_english(row):
    """
    Translate the 'title' column of a DataFrame row to English if the 'language' column is not 'en'.

    Args:
    - row (pd.Series): A row from a DataFrame.

    Returns:
    - pd.Series: The modified row.
    """
    if row['language'] != 'en':
        row['title'] = translate_to_english(row['title'])
    return row



def create_dataset():
    """
    Create a dataset that contains pairs ((img, text), (img, text)) of ads with a label indicating similarity.

    Args:
    - df (pd.DataFrame): The input DataFrame containing ad data.

    Returns:
    - pd.DataFrame: The created dataset.
    """

    from utils.constants import DATA_PATH


    df = pd.read_csv(DATA_PATH + "data.txt", sep=';', on_bad_lines='skip', names=['cc', 'title', 'turl', 'label'])

    # Rename images in folder, this is supposed to run only once
    image_folder = DATA_PATH + 'images'
    rename_images(image_folder)



    # Text preprocessing
    df = df[~((df['label'] == 193) & (df['cc'] == '1777'))]


    df['language'] = df['title'].apply(lambda x: detect_language(x))
    df = df.dropna().reset_index(drop=True)
    df['label'] == df['label'].astype(int)


    # Apply the translation function to the DataFrame
    # df = df.apply(translate_if_not_english, axis=1)


    del df['language']
    df['img_identifier'] = df.apply(create_identifier, axis=1)


    del df['cc']
    del df['turl']

    # Creating a copy of the original dataset to formulate the pairs.
    # Since pairs should be two separate ads, shuffling is utilized
    df2 = df.copy()
    df = df.sample(frac=1).sort_values(by='label').reset_index(drop=True)
    df2.rename(columns={'img_identifier': 'img_identifier_2', 'title': 'title_2'}, inplace=True)
    df2 = df2.sample(frac=1).sort_values(by='label').reset_index(drop=True)

    # Similar (or duplicate) ads between the two dataframes have the same label
    similar = pd.concat([df, df2], axis=1)
    del similar['label']
    similar['label'] = 1


    # Similar (or duplicate) ads between the two dataframes have different labels
    df2 = df2.sort_values(by='label', ascending=False).reset_index(drop=True)
    non_similar = pd.concat([df, df2], axis=1)
    del non_similar['label']
    non_similar['label'] = 0


    whole = pd.concat([similar, non_similar])
    whole = whole.sample(frac=1).reset_index(drop=True)
    whole.to_csv(DATA_PATH + 'data.csv', index=False)
    return 'dataset created'
