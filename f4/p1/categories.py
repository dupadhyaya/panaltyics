# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 06:26:48 2017 by Dhiraj Upadhyaya
Category 
https://pandas.pydata.org/pandas-docs/stable/categorical.html#categorical-rfactor
"""
import pandas as pd
s = pd.Series(["a","b","c","a"], dtype="category")
s

df = pd.DataFrame({"A":["a","b","c","a"]})

df["B"] = df["A"].astype('category')

df.dtypes
df.describe
df.shape

# Freq Distribution
df = pd.DataFrame({'value': np.random.randint(0, 100, 20)})
df
labels = [ "{0} - {1}".format(i, i + 9) for i in range(0, 100, 10) ]
labels
# cut and breaks
df['group'] = pd.cut(df.value, range(0, 105, 10), right=False, labels=labels)

df.head(10)

raw_cat = pd.Categorical(["a","b","c","a"], categories=["b","c","d"], ordered=False)
raw_cat
s = pd.Series(raw_cat)
s

df = pd.DataFrame({"A":["a","b","c","a"]})
df["B"] = raw_cat
df

s = pd.Series(["a","b","c","a"])
s
s_cat = s.astype("category", categories=["b","c","d"], ordered=False)
s_cat
df.dtypes

#In contrast to R’s factor function, categorical data is not converting input values to strings and categories will end up the same data type as the original values.
#In contrast to R’s factor function, there is currently no way to assign/change labels at creation time. Use categories to change the categories after creation time.
#o get back to the original Series or numpy array, use Series.astype(original_dtype) or np.asarray(categorical):
s = pd.Series(["a","b","c","a"])
s
s2 = s.astype('category')
s2
s2.astype(str)
np.asarray(s2)

#If you have already codes and categories, you can use the from_codes() constructor to save the factorize step during normal constructor mode:
splitter = np.random.choice([0,1], 5, p=[0.5,0.5])
s = pd.Series(pd.Categorical.from_codes(splitter, categories=["train", "test"]))
s

#Describe
#Using .describe() on categorical data will produce similar output to a Series or DataFrame of type string.
cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
df = pd.DataFrame({"cat":cat, "s":["a", "c", "c", np.nan]})
df.describe()
df["cat"].describe()

#Categorical data has a categories and a ordered property, which list their possible values and whether the ordering matters or not. These properties are exposed as s.cat.categories and s.cat.ordered. If you don’t manually specify categories and ordering, they are inferred from the passed in values.
s = pd.Series(["a","b","c","a"], dtype="category")
s.cat.categories
#Index(['a', 'b', 'c'], dtype='object')
s.cat.ordered

#New categorical data are NOT automatically ordered. You must explicitly pass ordered=True to indicate an ordered Categorical.
#The result of Series.unique() is not always the same as Series.cat.categories, because Series.unique() has a couple of guarantees, namely that it returns categories in the order of appearance, and it only includes values that are actually present.
s = pd.Series(list('babc')).astype('category', categories=list('abcd'))
s
s.cat.categories
s.unique()

#Renaming Categories
#Renaming categories is done by assigning new values to the Series.cat.categories property or by using the Categorical.rename_categories() method:
    
s = pd.Series(["a","b","c","a"], dtype="category")
s
#new
s.cat.categories = ["Group %s" % g for g in s.cat.categories]
s
#Categories (3, object): [Group a, Group b, Group c]
#rename
s.cat.rename_categories([1,2,3])
#Categories (3, int64): [1, 2, 3]

#In contrast to R’s factor, categorical data can have categories of other types than string.
#Be aware that assigning new categories is an inplace operations, while most other operation under Series.cat per default return a new Series of dtype category.

#Categories must be unique or a ValueError is raised:
"""
try: 
    s.cat.categories = [1,1,1]
    except ValueError as e:
        print("ValueError: " + str(e)) 
"""
#Categories must also not be NaN or a ValueError is raised:

#Add new Categories
#Appending categories can be done by using the Categorical.add_categories() method:
s = s.cat.add_categories([4])
s.cat.categories
#Index(['Group a', 'Group b', 'Group c', 4], dtype='object')
s
#Categories (4, object): [Group a, Group b, Group c, 4]

#Removing Categories
#Removing categories can be done by using the Categorical.remove_categories() method. Values which are removed are replaced by np.nan.:
s = s.cat.remove_categories([4])
s
#Categories (3, object): [Group a, Group b, Group c]
s

#Removing Unused Cateogories
#Removing unused categories can also be done:
s = pd.Series(pd.Categorical(["a","b","a"], categories=["a","b","c","d"]))
s
s.dtype
s.cat.remove_unused_categories()
#Categories (2, object): [a, b]

#Setting Categories
#If you want to do remove and add new categories in one step (which has some speed advantage), or simply set the categories to a predefined scale, use Categorical.set_categories().

s = pd.Series(["one","two","four", "-"], dtype="category")
s
s.dtype

s = s.cat.set_categories(["one","two","three","four"])
s
#Categories (4, object): [one, two, three, four]

#Be aware that Categorical.set_categories() cannot know whether some category is omitted intentionally or because it is misspelled or (under Python3) due to a type difference (e.g., numpys S1 dtype and python strings). This can result in surprising behaviour!

#Sorting and Order
#If categorical data is ordered (s.cat.ordered == True), then the order of the categories has a meaning and certain operations are possible. If the categorical is unordered, .min()/.max() will raise a TypeError.

s = pd.Series(pd.Categorical(["a","b","c","a"], ordered=False))
s
s.sort_values(inplace=True)
s
s = pd.Series(["a","b","c","a"]).astype('category', ordered=True)
s
s.sort_values(inplace=True)
s
s.min(), s.max()
#You can set categorical data to be ordered by using as_ordered() or unordered by using as_unordered(). These will by default return a new object.
s.cat.as_ordered() #Categories (3, object): [a < b < c]
s.cat.as_unordered() #Categories (3, object): [a, b, c]

#Sorting will use the order defined by categories, not any lexical order present on the data type. This is even true for strings and numeric data:

s = pd.Series([1,2,3,1], dtype="category")
s
s = s.cat.set_categories([2,3,1], ordered=True)
s
s.sort_values(inplace=True)
s
s.min(), s.max()

#Reordering
#Reordering the categories is possible via the Categorical.reorder_categories() and the Categorical.set_categories() methods. For Categorical.reorder_categories(), all old categories must be included in the new categories and no new categories are allowed. This will necessarily make the sort order the same as the categories order.
s = pd.Series([1,2,3,1], dtype="category")
s
s = s.cat.reorder_categories([2,3,1], ordered=True)
s
s.sort_values(inplace=True)
s
s.min(), s.max()

#Note Note the difference between assigning new categories and reordering the categories: the first renames categories and therefore the individual values in the Series, but if the first position was sorted last, the renamed value will still be sorted last. Reordering means that the way values are sorted is different afterwards, but not that individual values in the Series are changed.
#Note If the Categorical is not ordered, Series.min() and Series.max() will raise TypeError. Numeric operations like +, -, *, / and operations based on them (e.g. Series.median(), which would need to compute the mean between two values if the length of an array is even) do not work and raise a TypeError.


#Multi Column Sorting
#A categorical dtyped column will participate in a multi-column sort in a similar manner to other columns. The ordering of the categorical is determined by the categories of that column.
dfs = pd.DataFrame({'A' : pd.Categorical(list('bbeebbaa'), categories=['e','a','b'], ordered=True), 'B' : [1,2,1,2,2,1,2,1] })
dfs
dfs.describe
dfs.sort_values(by=['A', 'B'])
dfs
#Reordering the categories changes a future sort.
dfs['A'] = dfs['A'].cat.reorder_categories(['a','b','e'])
dfs.sort_values(by=['A','B'])


# Comparisons
"""
Comparing categorical data with other objects is possible in three cases:

comparing equality (== and !=) to a list-like object (list, Series, array, ...) of the same length as the categorical data.
all comparisons (==, !=, >, >=, <, and <=) of categorical data to another categorical Series, when ordered==True and the categories are the same.
all comparisons of a categorical data to a scalar.
All other comparisons, especially “non-equality” comparisons of two categoricals with different categories or a categorical with any list-like object, will raise a TypeError.
"""
#Note Any “non-equality” comparisons of categorical data with a Series, np.array, list or categorical data with different categories or ordering will raise an TypeError because custom categories ordering could be interpreted in two ways: one with taking into account the ordering and one without.


