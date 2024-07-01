# Phrase Golf

Phrase Golf is a phrase guessing game which, rather than being based on 
letters like Wordle or Wheel of Fortune, it's based on semantic 
similarity (like Semantle). The way it works is, you guess any string of 
characters at all, and the game tells you how semantically related your 
guess is to the given target. You have an unlimited number of guesses, 
and when you get the target exactly right (100% similarity), you win.

Under the hood, Phrase Golf is based on a modern "text embedding model" 
(such as E5), which takes a piece of text as input and outputs a vector 
in an abstract 1024-dimensional space. Then we use the Cosine Similarity 
between two vectors to measure the semantic similarity between two 
pieces of text.
