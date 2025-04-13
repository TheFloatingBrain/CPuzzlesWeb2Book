# What is this?
This is a tool I created to scrapes C Puzzles from https://programmedlessons.org/CPuzzles/ and composes a programming PDF book (as an accompnying text for a C/other programming book) for a friend :) This was intended as a "quick little script" so its slightly hacky, a bit specific, and includes a few manual steps to make it work, thats okay, it puts a little love into the process ❤️ .

# License:
Please pay attention to the license, its unusual (its an Ethical Source License), and has potential implications for the output.

# Outline:
 - First the script can be used to download all the Question, Answers, and info/intro pages and associated images from https://programmedlessons.org/CPuzzles/ (`download:special`, `download:images`, `download:qa` commands).
 - Second it uses Beautiful Soup 4 to divide the question pages into "top" containing some info, and each question into its own seperate page, called a "segment" (`generate:qa_segments` comamnd)
 - Third it can generate "Chapter" pages, an Index and include a custom cover/forward page (called C Puzzlers.pdf) into a custom Preamble (`generate:chapter_pages`, `assemble:preamble`)
 - Segments are assembled into Sections and Sections + other pages into a book (`assemble:segment`, `assemble:chapters`, `assemble:book`, `assemble:book_contents`). 

# Future:
 I am considiring getting the author's permission to publish this as a book, I would also potentially like to do some of the other web tutorials, and publish them as well. For now its just this Im publishing.


