#!/usr/bin/env python
# coding: utf-8

# In[1]:


def count_and_print(n):
    """Count from 1 to n whilst displaying a message

    Count from 1 to an integer n (inclusive). For each count, print the 
    counter value and a message separated by a single space. If the counter is
    even, the message should be `fizz`, `fuzz` if divisible by 3 and `foo` 
    (only) if divisible by 6.

    Parameters
    ----------
    n : integer
        Count up to this number (inclusive)

    Examples
    --------
    >>> count_and_print(12)
    1
    2 fizz
    3 fuzz
    4 fizz
    5
    6 foo
    7 
    8 fizz
    9 fuzz
    10 fizz
    11
    12 foo

    """
    i = 1
    msg = ""
    while i <= n:

        if i % 2 == 0 and i % 6 != 0:
            msg = "fizz"
            print(f"{i}" + f" {msg}")
        elif i % 3 == 0 and i % 6 != 0:
            msg = "fuzz"
            print(f"{i}" + f" {msg}")
        elif i % 6 == 0:
            msg = "foo"
            print(f"{i}" + f" {msg}")
        else:
            print(f"{i}")

        i += 1


# In[2]:


def overlap_interval(start1, end1, start2, end2):

    if start1 > start2 and end1 > end2 and start1 < end2:  # case 1: 1 ahead of 2
        print(start1, end2)
    elif start1 < start2 and end1 < end2 and start2 < end1:  # case 2: 1 behind 2
        print(start2, end1)
    elif start1 < start2 and end1 > end2:  # case 3: 2 inside 1
        print(start2, end2)
    elif start1 > start2 and end1 < end2:  # case 4: 1 inside 2
        print(start1, end1)
    else:
        print("No overlap!")


# In[3]:


def factorial(n):

    x = 1
    for i in range(1, n + 1):
        x = x * i
    return x


def cosine(theta):
    """Return cosine of theta using Maclaurin's approximation

    Parameters
    ----------
    theta : float
        Value to compute cosine of, in radians

    Returns
    -------
    cos_theta : float
        Cosine of theta
    """
    total = 0
    n = 0
    while True:
        term = ((-1)**n) * (theta**(2 * n)) / factorial(2 * n)
        total += term
        n += 1
        if abs(term) < 10**(-15):
            break
    return total


# In[4]:


def gcd(a, b):

    while True:
        a = abs(a)
        b = abs(b)
        r = a % b
        a = b
        b = r
        if b == 0:
            break
    return a


# In[5]:


def biased_sum(*args, base=2):  # *args for dynamic arguments

    total = 0
    for i in args:
        if i % base != 0:
            total += i
        elif i % base == 0:
            total += 2 * i
    return total


# In[6]:


def last_in_sequence(digits):

    num = 0
    last = 0

    while not digits.find("0") == -1:

        for ind in range(0, len(digits)):
            if digits[ind] == str(num) and num < 9:
                last = num
                num += 1
            elif digits[ind] == str(num) and num == 9:
                last = num
                num = 0
        return last

    if digits.find("0") == -1:
        return None


# In[7]:


def check_password(password):

    password_list = []
    for ch in password:
        password_list.append(ch)

    lower_list = []
    upper_list = []
    num_list = []

    for val in password_list:
        if val.islower():
            lower_list.append(val)
        elif val.isupper():
            upper_list.append(val)
        elif val.isnumeric():
            num_list.append(val)

    if len(password_list) >= 8 and len(lower_list) >= 1 and len(
            upper_list) >= 1 and len(num_list) >= 1:
        return True
    else:
        return False


# In[8]:


def is_palindrome(text):

    text = text.replace(" ", "")

    text_inv = text[::-1]

    if text == text_inv:
        return True
    else:
        return False


# In[9]:


def create_squares(num_stars):

    top = "+ " + "- " * num_stars + "+ " + "- " * num_stars + "+" + '\n'
    topstar = "| " + "* " * num_stars + "| " + "  " * num_stars + "|"
    bottomstar = "| " + "  " * num_stars + "| " + "* " * num_stars + "|"
    for i in range(num_stars):
        topstar_mult = (topstar + '\n') * num_stars

    mid = "+ " + "- " * num_stars + "+ " + "- " * num_stars + "+" + '\n'
    for i in range(num_stars):
        bottomstar_mult = (bottomstar + '\n') * num_stars
    bottom = "+ " + "- " * num_stars + "+ " + "- " * num_stars + "+" + '\n'

    return top + topstar_mult + mid + bottomstar_mult + bottom


# In[10]:


def create_grid(num_squares, num_stars):

    top = "+" + (" -" * num_stars + " +") * num_squares + '\n'
    star = ""
    star2 = ""
    star_tot = ""

    for i in range(num_stars):
        if num_squares % 2 == 0:

            star = " *" * num_stars + " |" + "  " * num_stars + " |"
            star2 = "  " * num_stars + " |" + " *" * num_stars + " |"
            star_mult = ("|"+star*(num_squares//2)+"\n")*num_stars +                 "+"+(" -"*num_stars+" +")*num_squares+"\n"
            star_mult2 = ("|"+star2*(num_squares//2)+"\n") *                 num_stars + "+"+(" -"*num_stars+" +")*num_squares+"\n"
            star_tot = (star_mult + star_mult2) * (num_squares // 2)
        elif num_squares == 1:
            star_tot = ("| " + "* "*num_stars+"|\n")*num_stars +                 "+ "+("- "*num_stars+"+")*num_squares+"\n"
        else:
            star = " *" * num_stars + " |" + "  " * num_stars + " |"
            star2 = "  " * num_stars + " |" + " *" * num_stars + " |"
            star_mult = ("|"+star*(num_squares//2)+" "+"* "*num_stars+"|"+"\n") *                 num_stars + "+"+(" -"*num_stars+" +")*num_squares+"\n"
            star_mult2 = ("|" + star2 *
                          (num_squares // 2) + " " + "  " * num_stars + "|" +
                          "\n") * num_stars + "+" + (" -" * num_stars +
                                                     " +") * num_squares + "\n"
            star_tot = (star_mult + star_mult2) * (num_squares //
                                                   2) + star_mult

    return top + star_tot


# In[11]:


def chop(a_list):

    try:
        del a_list[0]
        del a_list[-1]
    except:
        return None
    return None


# In[12]:


def sum_multiples(a_list):

    sum_num = 0
    for i in a_list:
        if i % 3 == 0 or i % 5 == 0:
            sum_num += i

    return sum_num


# In[13]:


def rotate(numbers, k):
    """Rotate numbers by k elements"""
    new_numbers = []
    tot = len(numbers)
    for i in range(tot):
        ref_ind = (i - k) % tot
        new_numbers.append(numbers[ref_ind])
    return new_numbers


# In[14]:


def on_all(func, a_list):
    """Apply func to every element of a list"""
    new_list = []
    for val in a_list:
        y = func(val)
        new_list.append(y)
    return new_list


# In[15]:


def matrix_times_vector(mat, vec):
    
    product = []
    for row in mat:
        row_tot = 0
        for i in range(len(row)):
            term = row[i] * vec[i]
            row_tot += term
        product.append(row_tot)

    if not len(row) == len(vec):
        product = "Input dimensions are NOT compatible."
    return product


# In[16]:


def coder(text, to_morse=True):
    """Convert English text to Morse code and vice versa

    Parameters
    ----------
    text : str
        English text or Morse code text
    to_morse : boolean
        Converts text to Morse code if `True`, to English otherwise

    Returns
    -------
    coded_text : str
        Text converted to Morse code or English
    """
    tot = ""
    output = ""

    eng_to_morse = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        " ": " ",
        "": "",
        "/": "/"
    }

    morse_to_eng = {value: key for key, value in eng_to_morse.items()}

    if to_morse is True:  # english to morse
        text_cap = text.upper()
        for ch in text_cap:
            morse_char = eng_to_morse[ch] + " "
            tot += morse_char
        output = tot[0:-1]  # to remove the space at the end

    else:  # morse to english
        text_rep = text.replace("  ", " / ")
        text_list = text_rep.split(" ")
        for val in text_list:
            text_char = morse_to_eng[val]
            tot += text_char
        output = tot.replace("/", " ")

    return output


# In[17]:


def sort_by_key(items_with_keys, ascending=True):
    """Sort items_with_keys based on the keys then by item for same keys

    Parameters
    ----------
    items_with_keys : list of tuples
        list of (item, key) tuples to be sorted
    ascending : bool
        Sort in ascending order if True, in descending order otherwise
    """
    def numkey(val):  # this function returns the 2nd element per tuple
        return val[1]

    if ascending is True:
        # sorts by the 2nd element in ascending order
        items_with_keys.sort(key=numkey)

    else:
        # sorts by the 2nd element in descending order
        items_with_keys.sort(key=numkey, reverse=True)

    return items_with_keys


# In[18]:


def count_words(text):
    
    dic = {}

    split_text = text.lower().split()

    for key in split_text:
        dic[key] = split_text.count(key)

    return dic


# In[19]:


def display_tree(dictree, indent=""):

    output = ""
    sortree = sorted(dictree.items())
    for key, value in sortree:
        output += indent + str(key) + ":"

        if not type(value) is dict:  # not dict
            output += " " + str(value) + "\n"

        else:  # dict
            output += "\n" + display_tree(value, indent + "  ")

        print(output)

    return output


# In[20]:


def get_nested_key_value(nested_dict, key):
    
    key_split = key.split(".")
    dup_dict = nested_dict
    for i in key_split:
        try:
            dup_dict = dup_dict[i]
        except:
            return None
    return dup_dict


# In[21]:


def value_counts(a_list, out_path):

    out_list = []
    out = ""
    temp_list = []
    for val in a_list:
        out_list.append(a_list.count(val))

    pair_list = list(zip(a_list, out_list))

    pair_list.sort(key=lambda x: x[0])

    pair_list.sort(key=lambda x: x[1], reverse=True)

    for k in pair_list:
        if k not in temp_list:
            temp_list.append(k)
        else:
            temp_list = temp_list

    for val in temp_list:
        out += str(val[0]) + "," + str(val[1]) + "\n"

    f = open(out_path, "w")

    f.write(out)
    f.close()


# In[22]:


def is_subset(sublist, superlist, strict=True):
    """
    Check whether `sublist` is a subset of `superlist`

    Parameters
    ----------
    sublist : list
        List to check whether it is a subset of `superlist`
    superlist : list
        List to check whether `sublist` is one of its subsets
    strict : bool
        If `True`, the exact sequence of `sublist` must be found in 
        `superlist` for the `sublist` to be considered as a subset. If 
        `False`, `sublist` will be considered a subset of `superlist` as long
        as all members of `sublist` are found in `superlist`

    Returns
    -------
    is_subset : bool
        `True` is `sublist` is a subset of `superlist`, `False` otherwise
    """
    str_sublist = ""
    str_superlist = ""
    if strict is True:

        for val in sublist:
            str_sublist += str(val)

        for val2 in superlist:
            str_superlist += str(val2)

        return str_sublist in str_superlist

    elif strict is False:
        set_sublist = set(sublist)

        return set_sublist.issubset(set(superlist))


# In[23]:


def has_duplicates(a_list):

    return not len(a_list) == len(set(a_list))


# In[24]:


import pickle

def count_words(input_file, output_file):

    dic_pkl = {}

    f = open(input_file, "r")

    read_data = f.read()

    read_data = read_data.lower().split()

    for key in read_data:
        dic_pkl[key] = read_data.count(key)

    f2 = open(output_file, "wb")

    pickle.dump(dic_pkl, f2)


# In[25]:


import math

class Person:
    def __init__(self, c=(0, 0), infected=False):
        self.x = c[0]
        self.y = c[1]
        self.infected = infected

    def move(self, dx=0, dy=0):
        self.x += dx
        self.y += dy

    def get_position(self):
        return (self.x, self.y)

    def is_infected(self):
        return self.infected

    def set_infected(self):
        self.infected = True

    def get_infected(self, person, threshold):
        dist = math.sqrt((self.y - person.y)**2 + (self.x - person.x)**2)
        if person.infected is True and threshold > dist:
            self.infected = True


# In[26]:


class QuarantinedPerson(Person):
    
    def move(self, dx=0, dy=0):
        pass


# In[27]:


def file_lines(**kwargs):

    dic_files = {}

    for k, v in kwargs.items():

        try:
            with open(v, "r") as f:
                num_lines = len(f.readlines())
                dic_files[k] = num_lines

        except:
            pass

    return dic_files


# In[28]:


class TenDivError(ValueError):
    pass

def ten_div(n, d):

    try:
        if n >= 0 and n <= 10:
            q = n / d
            return q
        else:
            raise TenDivError()
    except Exception as e:
        raise TenDivError(f"Error encountered: {e}")

