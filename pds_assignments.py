#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

cat / mnt/data/public/gutenberg/1/6/6/1661/1661.txt | tail - n 21 | head - n 5


# In[2]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

cat / mnt/data/public/agora/Agora.csv | grep - w - i "Drugs/Cannabis/Weed" | wc - l


# In[3]:


get_ipython().run_line_magic('', 'bash')
rm - rf a
rm - rf d
rm - rf j

mkdir - p a/b/c d/e/f j/k/l


# In[4]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

ls - lh - lS / mnt/data/public/amazon-reviews | head - n 2 | tail - n 1


# In[5]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

find / mnt/data/public/millionsong/A - name * .h5 | sort


# In[6]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

find / mnt/data/public/census - size + 1000000c - ls | awk '{print $7,$11,$12,$13}' | sed 's/\\//g' | sort - n


# In[7]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

cat / mnt/data/public/gdeltv2/masterfilelist.txt | grep mentions | grep - E 2019021408 | cut - d " " - f 3


# In[8]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

cat / mnt/data/public/movielens/20m/ml-20m/tags.csv | cut - d "," - f 3 | sort - n | uniq | wc - l


# In[10]:


get_ipython().run_line_magic('', 'bash - -bg - -out output')

cat / mnt/data/public/book-crossing/BX-Books.csv | grep - w - i "pandemic" | cut - d ";" - f 2 > pandemic-books.txt


# In[12]:


def digit_sum(number):
    '''
    Takes a positive integer and returns the sum of its digits.
    '''
    str_list = []

    str_num = str(number)

    for i in str_num:
        str_list.append(i)

    num_list = [int(j) for j in str_list]

    return sum(num_list)


# In[13]:


def count_vowels(text):
    '''
    Returns the number of vowels in an input string.
    '''
    return(text.count("a") + text.count("e") + text.count("i") + text.count("o") + text.count("u"))


# In[14]:


def is_interlock(word_list, word1, word2):
    '''
    Check if word1 and word2 interlocks based on word_list

    Two words "interlock" if taking alternating letters from each forms a new 
    word. For example, "shoe" and "cold" interlock to form "schooled".


    Parameters
    ----------
    word_list : list
        List of valid words
    word1 : string
        First word to check
    word2 : string
        Other word to check


    Returns
    -------
    interlockness : bool
        True if `word1` and `word2` interlock
    '''
    word1_list = []
    word2_list = []
    intlock_list = []
    words = []

    for i in word1:
        if i.isalpha():
            word1_list.append(i)

    for j in word2:
        if j.isalpha():
            word2_list.append(j)

    if len(word1) > len(word2):
        for k in range(len(word2_list)):
            intlock_list.append(word1_list[k])
            intlock_list.append(word2_list[k])
        intlock_list.append(word1_list[k+1])
        words.append("".join(intlock_list))

    elif len(word1) < len(word2):
        for m in range(len(word1_list)):
            intlock_list.append(word2_list[m])
            intlock_list.append(word1_list[m])
        intlock_list.append(word2_list[m+1])
        words.append("".join(intlock_list))

    else:
        for n in range(len(word1_list)):
            intlock_list.append(word1_list[n])
            intlock_list.append(word2_list[n])
        "".join(intlock_list)
        words.append("".join(intlock_list))

        intlock_list = []

        for p in range(len(word1_list)):
            intlock_list.append(word2_list[p])
            intlock_list.append(word1_list[p])
        "".join(intlock_list)
        words.append("".join(intlock_list))

    for val in words:
        if val in word_list:
            return True
    return False


# In[15]:


def count_types(a_string):
    '''
    Accepts a string a_string, and returns the number of lowercase, uppercase,
    numeric, punctuation and whitespace characters in a_string as items
    in a dictionary.
    '''
    orig_len = len(a_string)
    no_newline = ""
    no_space = ""

    lowercase = 0
    uppercase = 0
    numeric = 0
    punctuation = 0
    white = 0

    dic = {}

    no_newline_list = a_string.strip().split("\n")

    no_newline = " ".join(no_newline_list)

    no_space_list = no_newline.split(" ")

    no_space = "".join(no_space_list)

    for i in no_space:

        if i.islower():
            lowercase += 1

        if i.isupper():
            uppercase += 1

        if i.isnumeric():
            numeric += 1

    punctuation = len(no_space) - lowercase - uppercase - numeric

    whitespace = orig_len - len(no_space)

    dic["lowercase"] = lowercase
    dic["uppercase"] = uppercase
    dic["numeric"] = numeric
    dic["punctuation"] = punctuation
    dic["whitespace"] = whitespace

    return dic


# In[17]:


def matmul(A, B):
    '''
    Accepts two matrices as list of lists and returns their matrix product as
    a list of lists.
    '''
    B_T = []

    P = []

    for j in range(len(B[0])):
        row_T = []
        for i in range(len(B)):
            row_T.append(B[i][j])
        B_T.append(row_T)

    for a in range(len(A)):
        row_P = []
        for b in range(len(B_T)):
            row_P.append(sum([x*y for x, y in zip(B_T[b], A[a])]))

        P.append(row_P)

    return P


# In[18]:


def encode(text):
    '''
    Takes in a a string message with no spaces between the words and returns
    the encode message.
    '''
    total = len(text)
    init = total**0.5

    code_array = []
    output = ""

    row_num = round(init)

    if row_num**2 >= len(text):
        col_num = row_num

    else:
        col_num = row_num + 1

    for i in range(col_num):
        code_array.append(text[i::col_num])

    output = " ".join(code_array)

    return output


# In[19]:


def check_brackets(str_with_brackets):
    '''
    Check whether str_with_bracks is bracketed correctly

    Parameters
    ----------
    str_with_brackets : str
        String with brackets that are possibly nested

    Returns
    -------
    is_correct : bool
        `True` if `str_with_brackets` is bracketed correctly, `False` 
        otherwise
    '''

    dicket = {"(": ")", "[": "]", "{": "}", "<": ">"}

    brackets = ""

    str_with_brackets = str_with_brackets.replace(
        "+", "").replace("-", "").replace("*", "").replace("/", "").replace(" ", "")

    for val in str_with_brackets:
        if not val.isalnum():
            brackets += val

    memory = ""

    for i in range(len(brackets)):
        if brackets[i] in dicket.keys():
            memory += brackets[i]
        elif dicket[memory[-1]] == brackets[i]:
            memory = memory[:-1]
        else:
            return False

    return True


# In[20]:


def nested_sum(list_of_lists):
    '''
    Takes a list of lists of integers and adds up the elements from all of the
    nested lists.
    '''
    total = 0

    for i in list_of_lists:

        if isinstance(i, list):
            total += nested_sum(i)

        else:
            total += i

        print(total)
    return total


# In[21]:


def count_people(log):
    '''
    Accepts a multiline string of log entries and returns the number of people
    inside the building based on the log.
    '''
    num_inside = 0

    log_list = ("\t".join(log.strip().split("\n"))).split("\t")

    for i in range(len(log_list)):
        if log_list[i] == "IN":
            num_inside += int(log_list[i+1])
        elif log_list[i] == "OUT":
            num_inside -= int(log_list[i+1])

    return num_inside


# In[22]:


from collections import Counter


def next_word(text, word=None):
    '''
    Return the most likely next word in text

    A word is defined as a sequence of all non-whitespace characters between
    whitespaces. Words are case-insensitive.

    Parameters
    ----------
    text : string
        Text to train at.
    word : string or `None`
        Find the most likely next word of `word` or likely next word of all
        words if `None`. 


    Returns
    -------
    next_word : tuple or list of tuple
        If `word` is a string then return the most likely next word of `word`
        as a tuple of `(word, most_likely_next_word)`. If `word` is not found
        in `text`, `most_likely_next_word` is an empty string. If `word` is
        `None` then return the list of `(word, most_likely_next_word)` for all
        words in `text`. If there is more than one most likely next word, pick
        the first word based on alphabetical (lexicographic) order.
        '''

    text = text.lower().split(" ")

    dic_count = {}

    set_text = set(text)

    for set_word in set_text:

        occur = []
        count_occur = []

        for i, val in enumerate(text):
            if set_word == val:
                try:
                    occur.append(text[i+1])
                except:
                    pass

        dict_occur = dict(Counter(occur))

        count_occur = [(k, v) for k, v in dict_occur.items()]

        count_occur.sort(key=lambda x: x[0])

        count_occur.sort(key=lambda x: x[1], reverse=True)

        dic_count[set_word] = count_occur[0][0]

    if word is None:
        final_list = [(key, val) for key, val in dic_count.items()]
        final_list.sort(key=lambda x: x[0])
        return final_list

    else:
        return (word, dic_count[word])


# In[24]:


def div(a, b):
    '''
    Accepts a and b as parameters and returns a / b but will handle errors.
    '''
    if b == 0:
        return np.nan
    else:
        return a / b


# In[25]:


def gen_array():
    '''
    Returns a 10 × 10 numpy ndarray from numbers 1 to 100 with values 
    divisible by 3 replaced by 0.
    '''
    a = np.array(range(1, 101)).reshape(10, 10)
    a[a % 3 == 0] = 0
    return a


# In[26]:


def dot(arr1, arr2):
    '''
    Takes in two 1 × 9 numpy ndarrays, converts the arrays into 3 × 3 
    (e.g., [1, 2, 3, 4, 5, 6, 7, 8, 9] becomes
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]) then returns the dot product
    of the arrays.
    '''
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        raise ValueError
    elif (arr1.shape != (1, 9)) or (arr2.shape != (1, 9)):
        raise ValueError
    else:
        arr1 = arr1.reshape(3, 3)
        arr2 = arr2.reshape(3, 3)
    return np.matmul(arr1, arr2)


# In[27]:


def mult3d(a, b):
    '''
    takes in a 3D ndarray a and a square ndarray b and returns an ndarray z 
    defined by  𝑧𝑖𝑗𝑘=𝑎𝑖𝑗𝑘𝑏𝑖𝑗.
    '''
    b = b[:, :, np.newaxis]
    return np.multiply(a, b)


# In[28]:


class ABM:
    def __init__(self):
        self.timestep = 0

    def step(self):
        self.timestep += 1

    def status(self):
        return 'step'


def step_model(model, steps):
    statuses = []
    for _ in range(steps):
        statuses.append(model.status())
        model.step()
    return statuses


class ABMWithStep(ABM):
    def status(self):
        out = f"step {self.timestep}"
        return out


# In[29]:


class Tracker:
    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def get_position(self):
        return (self.lon, self.lat)


class FlightTracker(Tracker):
    def __init__(self, lon, lat, height):
        super().__init__(lon, lat)
        self.height = height

    def get_height(self):
        return self.height


# In[30]:


class Polygon:
    def __init__(self, *sides):
        self.sides = sides

    def compute_perimeter(self):
        return sum(self.sides)


class Rectangle(Polygon):
    def __init__(self, length, width):
        self.sides = [2 * length, 2 * width]


class Square(Rectangle):
    def __init__(self, side):
        self.sides = [side, side, side, side]


# In[31]:


f = open("transition.py", "w")
lines = """import numpy as np
class TransitionMatrix():
    def __init__(self, array):
        if np.any(array < 0) or np.any(array > 1):
            raise ValueError
        elif not isinstance(array, np.ndarray):
            raise TypeError
        else:
            self.array = array
            self.probabilities = array

    def step(self):
        array = TransitionMatrix(self.array)
        array.probabilities = self.probabilities * self.array
        return array"""
f.write(lines)
f.close()


# In[32]:


def peel(df):
    '''
    Accepts a square data frame df and returns a data frame with the
    outer square removed.
    '''
    df1 = df.copy()
    df1 = df1.iloc[1:-1, 1:-1]
    return df1


# In[33]:


def patch(df, upper_left, lst):
    '''
    Accepts a DataFrame df and modifies df by assigning the values of lst
    with the (0, 0) index of the list corresponding to upper_left.
    '''
    df.iloc[df.index.get_loc(upper_left[0]):(df.index.get_loc(upper_left[0]) + len(lst)), df.columns.get_loc(
        upper_left[1]):(df.columns.get_loc(upper_left[1]) + len(lst[0]))] = [[i for i in j] for j in lst]
    return df


# In[34]:


def pop_stats(province, municipality=None, census_year=2015):
    '''
    Accepts three parameters: province (required), municipality (optional)
    and census_year (defaults to 2015) and reads Municipality Data - PSA.csv.

    1. Returns the population for the given municipality if municipality
    is given, for that census_year.
    2. Returns the total population, mean population, and unbiased standard
    deviation for that province on that census_year if municipality is not given.
    3. Returns None when the given province, municipality, and/or census_year is
    not available.
    '''
    df = pd.read_csv("Municipality Data - PSA.csv")
    df.columns = df.columns[:2].tolist() + pd.to_datetime(df.columns[2:-1],
                                                          utc=True, format="%b-%y").year.tolist() + df.columns[-1:].tolist()
    df.rename(columns={2060: 1960}, inplace=True)
    if (not df["province"].str.contains(province.title()).any()) | (census_year not in df.columns.values):
        return None
    elif (municipality is None) & (df["province"].str.contains(province.title()).any()):
        df2 = df.loc[df["is_total"] == 0]
        pop_agg = df2.groupby("province")[census_year].agg(
            ["sum", "mean", "std"])
        return pop_agg.loc[province.title(), :]
    elif df["municipality"].str.contains(municipality.upper()).any():
        pop = df.loc[df["municipality"] == municipality.upper()][census_year]
        return pop
    else:
        return None


# In[35]:


def plot_pop(municipalities):
    '''
    Accepts a list of (case-insensitive) municipality names, reads
    Municipality Data - PSA.csv and returns a matplotlib Figure that
    replicates the figure below.
    '''
    df = pd.read_csv("Municipality Data - PSA.csv",
                     usecols=range(1, 12), index_col="municipality")
    df = df.loc[[m.upper() for m in municipalities]]
    ax = df.T.plot()
    ax.legend(municipalities)
    plt.xticks(rotation=45)
    return ax


# In[36]:


def find_max(province):
    '''
    Accepts a province (case-insensitive) and reads
    Municipality Data - PSA.csv then returns the municipality and census dates
    where the change in population is greatest.
    '''
    df = pd.read_csv("Municipality Data - PSA.csv")
    df = df.loc[df["is_total"] == 0]
    if df["province"].str.contains(province.title()).any():
        df = df.loc[df["province"] == province.title()]
        df2 = df.loc[:, "May-70":"Aug-15"]
        df2 = df2.diff(axis=1)
        max_diff = df2.max().sort_values(ascending=False).tolist()[0]
        max_diff_yr2 = df2.max().sort_values(ascending=False).index.tolist()[0]
        mun = df.loc[df2[max_diff_yr2] == max_diff]["municipality"].squeeze()
        max_diff_yr1 = df.columns[df.columns.get_loc(max_diff_yr2) - 1]
        return mun, max_diff_yr1, max_diff_yr2
    else:
        raise ValueError


# In[37]:


def most_populous():
    '''
    Reads Municipality Data - PSA.csv and returns a pandas Series of the 10 
    provinces with the most mean population for Aug-15.
    '''
    df = pd.read_csv("Municipality Data - PSA.csv")
    df = df.loc[df["is_total"] == 0]
    df = df.iloc[:, [0, -2]]
    sr = df.groupby("province")["Aug-15"].mean().sort_values(ascending=False)
    return sr


# In[38]:


def hourly_hashtag():
    '''
    Reads the first 1M data lines of
    /mnt/data/public/nowplaying-rs/nowplaying_rs_dataset/user_track_hashtag_timestamp.csv
    and returns a pandas data frame with columns hashtag, created_at and count.
    '''
    df = pd.read_csv("/mnt/data/public/nowplaying-rs/nowplaying_rs_dataset/user_track_hashtag_timestamp.csv",
                     nrows=1_000_000, usecols=["hashtag", "created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["created_at"] = df["created_at"].dt.tz_convert("Asia/Manila")
    df2 = df.groupby(["hashtag", pd.Grouper(key="created_at",
                     freq="H")]).size().reset_index(name="count")
    return df2


# In[39]:


def aisle_counts():
    '''
    Reads the first 1M data lines of 
    /mnt/data/public/instacart/instacart_2017_05_01/order_products__prior.csv
    and returns a pandas Series of the number of orders per aisle sorted by
    decreasing number of orders.
    '''
    df = pd.read_csv("/mnt/data/public/instacart/instacart_2017_05_01/order_products__prior.csv",
                     usecols=["order_id", "product_id"], nrows=1_000_000)
    df2 = pd.read_csv("/mnt/data/public/instacart/instacart_2017_05_01/products.csv",
                      usecols=["product_id", "aisle_id"])
    df3 = pd.read_csv(
        "/mnt/data/public/instacart/instacart_2017_05_01/aisles.csv", usecols=["aisle", "aisle_id"])
    df4 = (df.merge(df2, on="product_id", how="left")).merge(
        df3, on="aisle_id", how="left")
    sr = df4.groupby("aisle")["order_id"].count().sort_values(ascending=False)
    return sr


# In[40]:


def from_to():
    '''
    Reads the first 1000 data lines of
    /mnt/data/public/wikipedia/clickstream/clickstream/2017-11/clickstream-enwiki-2017-11.tsv.gz
    and returns a pandas DataFrame where the index are the unique values of
    the source column (first column) sorted in lexicographical order,
    the columns are the unique values of the destination column (second column)
    sorted in lexicographical order and the values are the corresponding views
    (fourth column).
    '''
    df = pd.read_csv("/mnt/data/public/wikipedia/clickstream/clickstream/2017-11/clickstream-enwiki-2017-11.tsv.gz",
                     nrows=1_000, sep="\t", header=None, compression="gzip", engine="python")
    df = df.rename(columns={0: "source", 1: "destination", 3: "views"})
    df = df.drop(columns=2)
    df = df.pivot(index="source", columns="destination", values="views")
    df.fillna(0, inplace=True)
    return df


# In[ ]:




