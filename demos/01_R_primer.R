

# RStudio crash course -----------------------------------------------------------

    # This is the scripting window... think of it like a notepad or word document, except it
    # ends in a ".R" extension instead of a ".txt" or a ".docx" like notepad / word respectively
    
    
    # Anything that starts with a pound-sign / hash / hash-tag / octothorp (yes that's a real term) is a comment
    # Both of these lines are comments and they won't be looked at when running the script
    
    
    # in order to run a single line of code at a time, hold "CTRL" and press "ENTER"
    # if there is a line of code (or a code block) to execute, then it will be executed and returned in the console below
    
    2 + 5  # put the cursor on this line, hold "CTRL" and press "ENTER" -- then check the console below
    
    print("hello world")
    
    
    # hold "CTRL" and press the "L" key to clear the console
    # The console is just temporary output, so that won't affect the script at all


# Variables ------------------------------------------------------------------------
    

    # Store values in variables with the "<-" syntax
    
    # here, we're storing the value 5 in the variable (or placeholder) of x:
    x <- 5
    
    # if you executed the line above, you should now see "x" in the top right window.
    # that is our environment, it keeps track of stuff we've initialized...
    
    # we can now reference x whenever we want, and it'll know we really mean 5
    print(x)
    
    x + 2


# data / class types ----------------------------------------------------------------
    
    
    # use the "class()" function to determine what type of object "x" really is:
    class(x)
    
        #' Sidenote: "class()" is referred to as a function. We can pass objects
        #' into functions if they are set up in a way to deal with those objects.
        #' Things we pass into functions are called "arguments"
        #' 
        #' In the context above, x is an argument to the class function. The class
        #' function is using the object x to determine what type x is, then returning
        #' the value of the data type of x back to us.
    
    
    # x is numeric because R was smart enough to know that's what we probably meant
    # let's check out some other data types
    my_text <- "hello world"  # saving the value "hello world" into a variable named my_text
    class(my_text)
    
    
    # my_text is a "character" object which can hold text
    # we can accomplish quite a bit with just these two data types

    # a third data type that is important to know about is the "logical" data type
    # logical data types take the form of either TRUE or FALSE
    # these are commonly referred to as "boolean" data types in other languages
    
    1 == 2   # FALSE
    7 == 7   # TRUE
    
    # these can also be stored
    my_logical <- my_text == "hello world"
    my_logical
    
    # logicals are critical for control flow (we'll explore that a little later) and
    # for subsetting larger data types such as vectors, lists, and data.frames
        

# Vectors ------------------------------------------------------------------
    
    # using single variables and values is great, but very often we need to be able
    # to store and process groups of values in the same object. That is where vectors 
    # and lists come into play
    
    test_grades <- c(84, 72, 97, 88)
    
    # the c() function stands for "combine," as it will group the values passed to it
    # into a vector. 
    
    # for any function that we don't define ourselves, we can find a meaningful explanation
    # of that function by typeing a "?" before it and executing it either in our script
    # or in the console itself
    ?c   # executing this line should open a help window in the bottom right quarter of the screen
    
    # many expressions in R are "vectorized" - meaning they will apply to every element
    # within the vector
    test_grades > 90  # FALSE FALSE TRUE FALSE
    
    # the sole "TRUE" value above corresponds to the third element in our test_grades
    # Note: this collection of logicals that is returned to us is also a vector itself
    
    # we can use [angle brackets] to subset a vector either by index value or with
    # logicals themselves
    
    # by index:
    test_grades[3]
    
    
        # sidenote: you can create sequences of integers if you put a colon (:) between them
        1:10    # 1 2 3 4 5 6 7 8 9 10
        
        # this is useful for subsetting as well
        test_grades[1:3]    #  first three test grades
    
    
    # by logical vector:
    test_grades[ test_grades > 85 ]
    
    # that was a bit of a jump, let me explain that a bit more clearly
    # when subsetting, a "TRUE" value means we should keep that item that lines
    # up with that "TRUE"... we'll then remove all of the "FALSE" values
    # So our subset expression above is the same thing as manually passing in 
    # a vector of logicals
    test_grades[ c(FALSE, FALSE, TRUE, TRUE) ]
    
    
    # we can also save this vector of logicals into it's own object and THEN use it
    # to subset the original vector, like so...
    
    good_grades <- test_grades > 85
    test_grades[good_grades]
          
    
    # these are the basic building blocks of what makes up complex data manipulation
    
    
# lists ------------------------------------------------------------------------------
    
    # vectors all had to be made up of the same type of data. All numeric; All character; etc.
    # lists are very similar to vectors, except the contents of a list can be a mix of all
    # kinds of different object types
    my_list <- list(5, "yep", TRUE, "this is fine", 3.14)
    my_list
    
    # you can even put lists inside of lists
    # Yo dawg, I heard you like lists, so I put some lists in your lists 
    nested_lists <- list(list(1, 2, 3), list(4, 5, 6))
    nested_lists
    
    
    # this gets confusing pretty quickly
    # this is also one of the areas where Python is cleaner than R
    # Python allows for the "Dictionary" data type which I think is a 
    # better way to handle nested data than lists of lists
    
    

# tabular data -----------------------------------------------------------------------
    
    # Most of you are familiar (to say the very least) with Excel.
    # No data / BI tool would be complete without some way to handle tabular data
    # much like a spreadsheet does in Excel.
    
    # In R, that object type is a data.frame (data.table was later created to extend
    # the functionality of data.frames, but that is outside the scope of this class)

    ?data.frame
    
    df <- data.frame(
        # think of each one of these as a column of data in a table
        id              = c(1, 2, 3),
        hours_of_sleep  = c(5, 9, 7.5),
        day_of_week     = c("mon", "tues", "wed"),
        test_score      = c(79, 83, 94),
        
        stringsAsFactors = F  # most annoying thing about data.frames, you usually want this FALSE
    )
        
    
    # a lot of people get pretty heated about the stringsAsFactors argument when constructing data.frames:
    # https://simplystatistics.org/2015/07/24/stringsasfactors-an-unauthorized-biography/ 
    

    print(df)

    
    # subsetting follows similar rules for data.frames
    df[2, 4]   # "give me the element in the second row of the fourth column"
        

    
    
# write your own functions --------------------------------------------------------------

    # it is also very useful to be able to write your own functions
    
    # I'm using the paste function in my custom function, so I'll explain that briefly here
    paste("this is a string", "AND THIS IS A STRING")
    
    # paste just combines two character strings
    
    # we can also manually specify what to separate the strings with using the "sep" argument
    paste("this is a string", "AND THIS IS A STRING", sep="----")
    
    
    greet <- function(name) {
        this_greeting <- paste("Greetings ", name, "!", sep='')
        return(this_greeting)
    }
    
    
    greet("Taylor")
    greet("BI Class")
    
    
# loading libraries (code smart people have written) ----------------------------------------------
    
    library(dplyr)
    
    # type the name of the library we loaded in, then two colons (::) to see what all
    # you can do with the package. Note, this also works for packages that already exist
    # in R without having to load them in, ie the "base" and "stats" packages
    
    # uncomment these lines that end in the double colon
    # don't execute these, just type the second colon and browse
    # base::  # this is a good way to learn what all is available in base R
    # stats:: # same with this

    # now for the package we just loaded
    # dplyr::

    # this double colon syntax is also useful for resolving name-space issues as well
    # If I load in two packages that each specify a "filter()" function, I can use
    # "dplyr::filter()" to tell R that I want to use the one that comes from the dplyr package
    base::paste("string1", "string2", sep=' - - - - - ')
    
    names(df)
    
    dplyr::filter(df, hours_of_sleep < 7)
    
    # insight: get more hours of sleep on sunday nights
    

# remove items from environment ---------------------------------------------------------------------
    
    #' to clear your environment, you can either click the little broom / paint-brush-looking thing
    #' in the top right quarter of the screen, or you can run the rm() function
    
    ?rm
    
    rm(df)   # remove a single item by name 
    
    ?ls      # this thing lists all the items in the environment
    
    
    # maybe we can combine these?
    
    rm(list = ls())   
    
        # We just passed what was returned from "ls()" to the argument of "rm()' called 
        # "list". That just removed every item in our environment so we have a clean
        # slate to work with. 
    
    # crash course is over, let's do something more fun
    
            
    
    
    


