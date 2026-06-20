# Speaker Script — Semantic Art Search
## Slide 1 — Title
"Hi, my project is a semantic art search engine. The short version: you type what you're looking for in plain English, 
and it finds matching paintings."

## Slide 2 — Problem Statement
"Normally, if you want to find a painting, you search by title or artist name. But that's not how people actually 
think about art — they think 'I want something moody and dark' or 'something with warm colors.' This project bridges 
that gap. You type a description, and the system understands it well enough to find the right paintings — that's the 
cross-modal part of this course: connecting language and vision."


## Slide 3 — Dataset
"I'm using 416 real paintings, pulled from Wikimedia Commons, split evenly across six art movements — impressionism, 
expressionism, surrealism, baroque, abstract, and romanticism. Each style sits in its own folder, which also gives me
free ground-truth labels later for evaluation."


## Slide 4 — Technical Pipeline
"There are three pieces working together. First, a computer vision model called SigLIP 2 turns every painting 
into a list of 1152 numbers — basically the model's internal understanding of what that image looks like. Second, 
the same model turns my text query into numbers in that exact same space. Third, I use cosine similarity — basically 
just measuring how close two sets of numbers are — to find which paintings are nearest to my query. That's the whole 
trick: text and images end up in the same mathematical space, so I can directly compare them."



## Slide 5 — Struggles 01: Model & Dataset Journey
"My first version used OpenCLIP and a general web-scraped dataset, and the results were honestly pretty bad — generic, 
mismatched. I didn't know if the problem was my data or the model itself. So I changed one variable at a time: first 
I swapped to a clean dataset, COCO, while keeping the same model — that didn't fix it. Then I kept COCO but swapped 
the model to SigLIP version 1 — that helped a lot. So the model mattered more than I expected. Eventually I landed 
on SigLIP 2 paired with this art-specific dataset, and that combination gave me the best results."



## Slide 6 — Struggles 02: Losing the Project
"At one point, after all these swaps, my project was scattered across three different folders, all half-broken in 
different ways. I actually lost track of which version was the 'real' one. I rebuilt it from scratch with one rule: 
every file has exactly one job, and there's a single file — models.py — that's the only place the model gets loaded. 
Everything else just imports from it. That made the whole thing much easier to debug and trust."



## Slide 7 — Struggles 03: Silent Bug + Model Choice
"This is probably my favorite bug story. At one point my indexing script was printing 'success' for every single 
image — but when I actually checked the database, most of the data wasn't there. It turned out a refactor had 
accidentally deleted the one line of code that actually saves the result to the database. The script was computing 
everything correctly and then just... throwing it away. The console log looked perfect the whole time, which is 
exactly why it took a while to notice — a clean-looking log isn't proof that something actually worked.

On the right side — picking a model wasn't simple either. There are dozens of vision-language models out there, 
and even published research doesn't fully agree on which is best. EVA-CLIP actually wins on some benchmarks, 
SigLIP 2 wins on others, like multilingual support. I'd like to test that head-to-head myself with more time, 
but for now I made a choice and validated it with my own evaluation instead of just trusting a published number."


## Slide 8 — Live System
"The actual app has three ways to explore the collection. Search lets you type something like 'beach at sunset' 
and get matching paintings. Browse Style lets you just look through one art movement at a time. And Similar Art 
lets you upload any image and find paintings that look visually similar to it — plus it'll guess what style your 
uploaded image looks most like."


## Slide 9 — Embedding Space & Evaluation
"This is where I actually checked whether the model learned anything meaningful. Every painting becomes a single 
point in this 1152-number space — way too many dimensions to look at directly, so I compressed it down to 2D just to 
visualize it. On the right, I ran a proper statistical test called a silhouette score, directly on the real 
high-dimensional vectors, not the simplified 2D picture. Baroque and surrealism scored the highest — meaning those 
styles are genuinely visually distinct. Abstract actually scored negative, which means abstract paintings in my 
dataset are, on average, closer to other styles than to each other. That's not really a flaw — it probably just means 
'abstract' is more of a conceptual label than one consistent visual look."


## Slide 10 — Evaluation Results (UPDATE WITH YOUR REAL NUMBERS)
"To check accuracy properly, I built a golden set — five queries where I manually verified, by actually looking at 
the images, which results were correct. My system got a Mean Average Precision of about 0.49, which is just under 
what's considered 'strong' retrieval, but solidly in the acceptable range. My recall was actually the weaker number 
— meaning some correct paintings exist in my collection but didn't always make it into the very top results. That 
tells me the ranking could be tuned further, even though the underlying embeddings clearly understand the content 
reasonably well."



## Slide 11 — Live Demo
"Now I'll just show you the actual thing running."
(Switch to your browser, demo Search, then Browse Style, then Similar Art, then Embedding Space tabs live.)


## Slide 12 — Next Steps
"A few honest next steps: I'd like to actually run that EVA-CLIP versus SigLIP 2 comparison instead of just citing 
other people's benchmarks. I'd like to grow the dataset past 416 images. I'd like to fine-tune the model specifically 
on art, since right now it's using general pretrained knowledge. And one more thing worth mentioning — nothing about 
this architecture is actually specific to art. The exact same code would work for searching industrial parts by photo, 
or real estate listings, or products in a catalog — the art dataset was just how I tested it. The architecture itself 
is general-purpose."


## Anticipated Questions — quick answers
**"Did the model learn the painting titles?"**
"No — the titles are just metadata pulled from the original filenames, sometimes in German since that's how 
Wikimedia Commons labeled them. The model never sees the title text at all — it only ever looks at the raw pixels."

**"How does indexing actually create the embedding?"**
"The image gets resized and normalized into the exact format SigLIP 2 expects, then it's run through the model's 
encode_image function, which outputs 1152 numbers — that's the embedding. It's not hand-crafted features, it's the 
output of a trained neural network."

**"Why is your overall silhouette score so low, only 0.073?"**
"Because art movements genuinely blend into each other historically. Expressionism grew out of reacting to 
impressionism, for example. A low-but-positive score is actually believable; a suspiciously perfect score 
would be the surprising result."

**"Why didn't you include the multimodal image+text combined query in the final demo?"**
"I built and tested it — it works correctly, I verified it mathematically — but I made a judgment call to cut it 
from the final UI because I couldn't argue for a strong enough real-world motivation for it in the time I had. 
It's listed as a next step rather than a finished feature."