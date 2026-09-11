# The Patriarchy Simulation — Script V3

> **Format guide**
> - `[VISUAL]` — animation / scene description
> - `[TEXT ON SCREEN]` — on-screen text overlays
> - *[stage direction]* — production / animation notes

---

## PART 1 — HOOK

This video is about NETWORKS, not biology

**[TIMER APPEARS]** Top right corner: `Simulation begins in  4:00` — small, white monospace font. Starts counting down.

Here is a research paper that graphs the iq differences between men and women: https://www.researchgate.net/publication/344751288_Scrutinizing_Distributions_Proves_That_IQ_Is_Inherited_and_Explains_the_Fat_Tail

*[show screenshot]*

The graph is simple. There are more men at the extreme ends of the spectrum. Women mostly occupy the middle. Concretely, what this means is that even though men and women have the same average iq, men are more likely to be geniuses or fools, while women are more likely to be average.

However, if you think about it, this graph does not really tell us anything useful at all. We already know that women are underrepresented in the extremes of society. But how do we know if this is due to biology, or due to the fact that society is just better at suppressing female behavior that is out of the ordinary? In other words, are our differences due to our nature, or our upbringing?

Scientific papers have raised multiple arguments on this matter.

*[show screenshot]*

But to be honest, none of them are conclusive. So lets answer a simpler question instead.

What if we simulated a world where men and women are completely equal in every way. Actually, what if we just simulated a society without gender at all. Can we still see gender differences?

---

## PART 2 — NETWORKS AND PATRIARCHY

This next section will contain questionable gender stereotypes. If you don’t agree with it, don’t worry, neither do I, they are just hypothetical examples to help build our intuition for the math that is about to follow

*[Giving examples before explaining the math]*

Let's start with the colour pink. Is there anything wrong with pink? Of course not. But then why is it that if you wear pink to a job interview, everyone will notice, but if you wear blue, even bright blue, nobody cares?

This isn’t specific to gender. Even a man wearing pink might be considered unprofessional.

But what if we start encouraging girls to like the colour pink from a young age? We have successfully created a small system of oppression. Let's take an even simpler example. Imagine two hobbies. One group spends every weekend watching football together. Another spends every weekend discussing fashion.

Neither hobby is objectively better

They just form 2 different networks

But what happens when one group has a much larger number of CEOs than the other.

Suddenly that network has internships, referrals, mentors and opportunities. This produces more CEOs and the cycle continues The hobby didn't create success The network did. And once that happens, everyone in that network benefits forever

The other network does not get these benefits to the same extent. Success is still possible, but it's just less probable.

Over time, people in each network begin to resemble one another. They don't just have different opportunities—they develop different cultures.

Now imagine if we groom boys and girls to have different hobbies from a young age.

They will develop shared interests that lead to further segregation later in life. This is terrible when we already have a starting inequality we want to fight

When society consists of multiple networks, how do we remove the inequalities of these networks?

What if 1 network is oppressed? How do we make that network more successful? Well, for a network to be successful, its people need to be successful, for the people to be successful, we need a successful network, for a network to be successful, we need the people to be successful, for people to be successful we need the ……

*[show 4 images in a circle of people becoming successful and networks becoming stronger]*

You get the idea. So lets refine our goal. If nobody discriminates… if men and women are born identical… And if society was completely fair But we have some starting inequality in our networks. What happens to our society over time? Do these inequalities get worse, or better Spoiler alert, its complicated. But there are solutions to this problem. We will be ranking them in a tier list, so stay tuned for that.

**[TIMER: ~2:30 remaining]**

---

## PART 3 — SIMULATION SETUP

**[VISUAL]** Black screen. A single dot appears. Then another. Rapid succession — 100 dots fill the screen

We are going to build a tiny society. One hundred people.

**[VISUAL]** Zoom into one dot. A profile card expands. Ten horizontal bars appear one by one, each a different length.

Each person has ten attributes. You can call them skills or traits — whatever. It doesn’t matter. For us, they are just numbers. *[first we show skills, then traits like confidence, then we just show attributes]*

Our society also has 3 different types of jobs, with different pays.

**[TEXT ON SCREEN]**

```
Job A  — 15 slots  — Pay: 1000
Job B  — 35 slots  — Pay: 300
Job C  — 50 slots  — Pay: 100
```

Remember each person’s set of attributes? Only the first 3 of them are important for getting a job.

We rank everyone by those three attributes. That ranking determines your job. Hiring is perfectly meritocratic — no one looks at anything other than the score.

So that’s the basic building blocks of our world. 100 people, 10 attributes, 3 jobs, and a perfectly fair hiring process. But hey, this is boring. Why is everyone identical? That’s not very interesting

So let's make a change. *[Visual: real attributes, innate attributes]*

Lets give everyone innate attributes. These are a set of hidden attributes that never change and represent each person's innate tendencies. Or their potential, if you like to think of it that way.

We will also give people influences. What is an influence? This is a set of attributes that could come from a book, a course, or an interaction with a stranger. *[show a set of attributes]*

People use these influences to update their attributes. *[show attributes overlapping and being added with influence]* However, not everyone learns the same way.

When people get an influence that aligns with their innate attributes, they learn very quickly. Similarly, people resist influence that goes against their innate attributes. This makes every person fundamentally unique.

So this is how our society works.

Time goes by in rounds.

Every round, 3 things happen *(this changes once we add networks)*:

1. People get a random influence *[update influence]*
2. People update their attributes based on their influence and their potential. *[update attributes]*
3. People are selected for jobs based on their updated attributes. *[update jobs]*

Remember, the people with the top values of attributes 1, 2 and 3 go to the best jobs.

And this happens over and over again. Many many times. This is how our simulation works.

*[keep showing 1, 2, 3 being coloured in order of execution with updating attributes, sound effect on every change, iteration counter]*

### Gender labels

Lets have some fun. We are going to have 2 secret genders in this world. The pinkies and the blueys.

*[show sets of grey dots side by side, then they get coloured]*

And now we are going to do something sneaky. We are going to secretly assign a gender to each person. And we aren't just going to assign genders at random. No, instead, we will just make most of the richest people blueys. They are the ones in Job A. The pinkies are mostly the ones who are in Job C. This is the starting distribution of genders among jobs.

*[show text]*

But nobody knows what their gender is, and it doesn’t have any other impact on our world.

At this point, you might say hey! Thats cheating. Just because people don’t know their gender, doesn’t mean that gender doesn’t exist.

But you see, thats the beauty of it. In the world we created, gender genuinely DOES NOT exist.

Previously, every person was identical. I just chose to give our inequalities a colour. You get it? The only reason we have gender at all is to keep track of how inequalities change.

Our world would be exactly the same if gender didn’t exist, but we would not be able to see how inequalities change with time. Its just our little secret.

**THERE ARE NO DIFFERENCES IN THE INNER POTENTIALS OF PINKIES AND BLUEYS.**

The blueys aren’t in Job A because they have the highest innate attributes for our job traits. They just happen to be the ones who happen to have the highest values today.

This means that eventually, the pinkies with the highest potentials should also join Job A, and we should start seeing equality between both genders. But will this actually happen?

Don't worry, the answer is complicated.

But before we get into the simulation, lets take a chill pill to gather our thoughts. *[peaceful music]*

**[TIMER hits 0:00 → disappears]**

**[TEXT ON SCREEN]** `Simulation starting.`

---

## PART 4 — TRIAL RUN

Lets do a small trial run.

Quick reminder: every round, people get a random influence, they update their attributes, and they get new jobs

Since people update their attributes based on their inner potential, this means that eventually people start drifting towards their innate attributes.

*[show real attributes slowly changing and becoming similar to innate attributes]*

And since pinkies and blueys have the same potential, we eventually see complete equality between them.

Most importantly, once equality forms, it stays forever. In our society, equality is a stable equilibrium.

If we roll a ball down a hill, it eventually settles at the bottom. Random changes cannot move it from its spot. This is a stable equilibrium. However, if we have a ball at the top of the hill, this is an unstable equilibrium. Any movement can take our system out of balance.

*[render and show on video]*

In this situation, patriarchy is the ball. The ball of patriarchy. It starts off on the left, with complete inequality. But eventually we find our way to the centre. Towards equality.

If this makes sense to you, congratulations. It's time to spice things up.

Lets start adding networks to our simulation.

---

## PART 5 — NETWORKS

So far, every influence was random. But thats not very realistic. In the real world, people also get influenced by their network.

the people in their job, and people they become friends with.

A person’s network exerts an influence on them. This is just the average attributes of everyone in their network.

But how do people become friends? friendships occur when there is a sufficiently high similarity score between 2 people.

The similarity score is just the cosine similarity of 2 sets of attributes in 10 dimensions. If that sentence scared you, don't worry. Just remember, when people are more than 80 percent similar, they become friends.

Why 80 percent? Because it's my video and I make the rules.

So now we are making a new change to our simulation. Previously, every person received a random influence. Now, people will receive 3 types of influences.

- A random influence
- An influence from their friends
- An influence from their job

So now every round, 4 things happen:

- People receive 3 sets of influences
- People update their attributes based on their influences
- People form new friendships based on their attributes
- People get selected into new jobs based on their first 3 attributes

What do you think will happen this time?

---

## PART 6 — SIMULATION 2: DYNAMIC FRIENDS

Would friendships lead to diversity? Turns out, it doesn’t work that way.

In this simulation, we NEVER achieve equality. No matter how far into the future we go, our gender pay gap NEVER goes away. *[show graph]* And this really is what I wanted to show you guys.

This was our old society. *[show ball on a hill]* Society kept moving towards equality.

However, our new society looks something like this. *[show a hill with 3 equilibriums, ie 3 valleys]*

When we start from an unequal world, we don’t reach equality. Instead, we reach another equilibrium where the blueys are in power forever. This is because the blueys eventually form a network that helps sustain their success.

The pinkies, however, do not get access to the same influences at all

Let's look at the story of one such pinkie: Emma

Emma is a really smart person. In fact, she has the highest potential among anyone else in our simulation. But she never moves out of Job C. How did this happen?

Her starting attributes were so different from the blueys that she never got opportunities to make the right connections to level up. In our trial run, she was able to make it to Job A using the random influences to level up. But this time, her random influences were always counter-acted by equally strong influences pulling her down. Emma, like many other pinkies, could never reach her full potential.

But hey, some pinkies are able to make it to Job A, so why can’t Emma?

Let's look at our friend groups. we have clusters of similar people forming friendships. However, there are occasional bridges, the in between friends, that connect all of them. These bridges allow influence to travel. However, the nature of our influences ensures that disjoint groups keep drifting further and further apart. the moment 2 groups become sufficiently different, the bridge between them goes away.

Let's take the example of Rachel — a pinkie who made it into Job B. She was lucky enough to get a few connections in Job B at the start. However, a few iterations after reaching Job B, she loses connections with her friends in Job C. She is no longer able to influence them. This includes Emma as well.

Eventually our society ends up forming disjoint echo chambers. And these echo chambers never allow the status quo to change. Some pinkies join bluey echo chambers, while some get left behind.

**[TIMER: ~2:30 — tier list begins]**

---

## PART 7 — OUR SOCIETY VS SOCIAL MEDIA

If this network feels familiar, that's because you already live inside one. Our current setup represents EXACTLY how social media works.

Social media algorithms are constantly collecting data on you based on the way you interact with content. *[show a node with hearts and thumbs up, down, sends, and changing attributes]* And using this, they give you a set of attributes — something that determines your personality.

If the algorithm determines a group of people to be similar enough, they now start influencing each other. *[show lines between them, clique]* Without even knowing it. Even 1 of these people liking a reel is enough for this reel to be fed to everyone in that group. *[1 person hearts → everyone hearts]* A person that is outside this similarity group will have almost no influence on these people. *[1 person outside dislikes]*

And here's the other things, modern algorithms are hypertuned to maximise engagement.

This means they are extremely efficient at cutting you off from content you may not enjoy. And they do this by cutting you off from groups that no longer match your interests This is how radicalization eventually occurs.

Here’s another problem we encountered. So far, we assumed everyone gets random influences every round. These really helped us fight our networks. But we now know that no content we consume on the internet is truly random. This means the only way to get truly random influences is to actually go out into the real world, and maybe meet a stranger on the street.

And that’s assuming you live on a diverse street.

Lets look at our society once again. *[show 3 equilibriums]* We see that equality is still an equilibrium. This means that as long as we can somehow push our starting state closer to equality, we should eventually be able to get there.

But first, a quick breather — chill out. *[peaceful music]*

---

## PART 8 — NOT A GENDER PROBLEM

At this point, it's pretty clear nothing we simulated is a gender problem. You could replace gender with race, sexuality, political preference, whatever, it wouldn't make a difference.

But you know what? At the risk of being controversial, lets MAKE this a gender problem

### Simulation 3 — Gender awareness

Our society is weird. Many people never reach their potential. But you can never know your potential unless you are given a chance.

Emma is perfectly happy in Job C. She doesn't know what she's missing. Financial ambition is one of the traits society has quietly suppressed in her.

So everyone lives happily. They believe the system is completely fair, and that they are responsible for their own outcomes.

Suddenly, we reveal our little secret. You all have a gender. It doesn’t control your abilities, and yet it controls your society.

People look around, and they notice that most of the wealthy people are blueys. In fact, they continue to be blueys, no matter what happens. For the first time, they are aware of inequality

The whole system is rigged! The poor say. The rich, however, claim: the system is completely fair, we never discriminated

In fact, some people in our world even conduct studies about it. Look, blueys have high job attributes. Surely this must mean something? Maybe pinkies just have a lower potential? Maybe there are biological differences between pinkies and blueys. Maybe someone is out there plotting the differences in IQ between pinkies and blueys.

But we are the gods of this world. And know that’s not true.

We created everyone equally, but some are just more equal than others.

Why don’t the pinkies just learn the attributes needed to become rich?

> What we see here isn't very different from the gender equality paradox in the real world. Why is it that richer, more gender-equal countries show larger variance in occupation between men and women?
>
> **In our world:** When both pinkies and blueys have the same freedoms, why do they continue having different outcomes? Don't we live in a gender-equal world? But having the same freedoms only means you are more free to act on your unequal influences.
>
> A poorer society would act as a common influence to both pinkies and blueys to prioritise job traits. But we don't simulate this pressure.

The question is, what are they going to do about it?

Lets find out, but feel free to take a break and think about it. *[peaceful music]*

---

## PART 9 — TIER LIST

Ok, now everyone knows about gender. And they know they have a problem

So they try making a bunch of simple changes to their system, and see which ones help fight our patriarchy.

I will be ranking their solutions in the form of a tier list

*(experiment 1/6)*

### Gender Quotas

Maybe reserving jobs for women will help?

Lets try and simulate this. Lets add a gender quota into our hiring process. For now, let's keep this quota small just to see what happens. 10 percent of positions in Job A are exclusively reserved for women. And for now, let's also assume no man discriminates against a woman. They will still form friendships with them at the same rate, and still allow them to compete for the remaining seats. What happens?

Sadly, a 10 percent reservation has no net effect on our system. We start out a little better, but we still end up at the same equilibrium. Why didn’t our equilibrium change?

Look at our previous society. Our equilibrium consisted of 20 percent pinkies in Job A. 10 percent isn’t enough to change that, the ball still lands in the same place

*[show ball starting at 10 and still landing at 21]*

**[TEXT ON SCREEN]**

```
No reservation:    Job A female — 21%   Pay: 43%
With reservation:  Job A female — 20%   Pay: 43%
(mean over 20 independent runs)
```
But lets take a closer look anyway

Remember Rachel? The pinkie who made it into Job B? This time, she gets into Job A due to reservation. But she is completely disconnected from all the male peers there. A 10 percent reservation also means that she is the only woman in Job A. In other words, she has no friends, and can’t use her position to impact society. By the time she can actually impact other women, our reservation has already become redundant.

Lets look at our society again. Maybe we just need to give more reservation, so that society actually moves to a new equilibrium

*[we have a graph where beyond 35% we start going towards a new equilibrium, the 2nd valley]*

Lets try giving a 40 percent reservation? The problem is, this changes the whole fabric of our world.

Earlier, equality was an equilibrium. But the moment we give 40 percent reservation, that equilibrium shifts.

In fact, with 40 percent reservation, The number of women in Job A keeps increasing until finally settling at around 82 percent. If we remove reservation then, our world once again shifts back, but this time, we end up with a matriarchy instead of a patriarchy. So now we have to solve the same problem in reverse.

So why not just remove reservation the moment there are 50 percent women in Job A? Does that lead to equality?

Since my society has some randomness, this sometimes works, but not always.

Think about it, once the gender pay gap goes to 0, does that mean we have achieved equality?

No, we haven’t, because women haven’t yet had time to remove the inequalities in their networks. They only have 50 percent representation because 40 percent was given for free. Their networks are still very different from men.

If we remove our quota the moment there are 50 percent women in Job A, we will once again become a patriarchy, and society corrects itself

In fact, finding the right balance of when to remove reservation is extremely difficult.

I tried running this simulation with multiple strategies, including a machine learning algorithm, and found it to be extremely fragile. You can never predict when you are heading towards equality or not.

When our underlying network takes us to equality on its own, we can just give it a push and let things be. This is what happens when our society automatically fights the patriarchy for us. But that is not the world we constructed here.

In this world, gender quotas start in the D tier — sometimes works, sometimes doesn’t.

*[show ball on a hill vs in a valley]*

By the way, something we never even captured.

The biggest problem with gender quotas, or any form of gender discrimination, is that men can also discriminate back. We assumed that everyone is a feminist, but real life doesn’t work that way. What if the act of adding quotas / reservations took us from a system that would have achieved equality on its own, to a system that is constantly fighting?

We humans are petty creatures. Think about it. When a man gets promoted, they definitely deserve it. But when a woman gets promoted, we question it. Was she placed there on merit, or for optics? And so we judge all women. Now women in Job A need to constantly prove they deserve to be there

Sorry, did I say women? I meant pinkies.

This is an objectively worse form of discrimination, and affects our networks as a whole.

Considering this, I am going to move it down to the F tier

Lets try another solution

*(experiment 2/6)*

### You Go Girl!

Ok, let's consider another simple change. What if the women help each other out of oppression?

Lets try simply reducing the similarity required for women to befriend other women. They are seeking out other female influences

Maybe they need other female role models to look up to.

Sadly, those role models don't exist in the first place, our simulation shows that this is terrible for society. By the time some rare women make it to Job A, they now have more male connections than female ones, despite the changes we made

*[show simulation result]*

We just added an inequality that made echo chambers much, much worse.

However, we can use the presence of role models along with reservation to get interesting results.

Look at the new state of our society. We moved the equilibrium even lower. This is why we end up in a worse state.

But crucially, we also reduced the amount of reservation needed to start changing society.

We can now achieve near equality with lower amounts of reservation, and then stop it when appropriate.

The reason I say near equality though, is because we never actually reach equality. We land at about 45 percent women in Job A

Lets look at our society once more, this time in a lot more detail: Previously, plenty of blueys and even pinkies were performing higher than their potential due to constantly getting only influences from a very successful echo chamber.

With all the changes we made, we now gave women a larger number of controlling influences from multiple different directions. They can now choose the influences that match their innate attributes.

This means that we actually REDUCED the peak performance of pinkies in the process, by normalising all of their traits, and taking them out of ultra successful echo chambers

But to be honest, I quite like the fact that our society allows its pinkies to only choose the influences they like. Maybe the blueys should do the same.

For that reason alone, I am now moving both You Go Girl and gender quotas back to the D tier. Although you can argue none of these things achieve equality at all.

*(experiment 3/6 — positive relationships)*

### Relationships

One major reason our fixes keep failing: we never built a way for pinkies and blueys to connect across networks. That was fine before gender was revealed — nobody was looking for it. But now everyone knows. What if they try to form heterosexual relationships?

And that's actually been a problem with our simulation. In real life, men and women don't just interact through friendships. They interact through relationships — male-female bonds we completely forgot about.

Relationships are interesting because they connect people who would never otherwise belong to the same network.

So let's add them to our world. Every person gets a partner at random. Every pinkie is paired with a bluey. And unlike friendships, relationships are allowed to form between completely different people.

What happens?

**Best case:** Maybe a relationship is just like a friendship. In a positive relationship, two people learn from each other. Maybe Emma falls in love with a guy who is doing really well at his job. So she gets better too. This is similar to how friends influence each other.

The relationship acts like a permanent bridge between two networks. And the result is ridiculous. We achieve equality in only 100 iterations.

Something I really like about this simulation is that everyone’s influences are extremely chaotic. Society is constantly pulling everyone in new and unique directions. Even though we reach equality pretty soon, people keep changing and reinventing themselves, hovering around their innate attributes.

**S tier.** We just got a beautiful, vibrant society. We didn’t even have to do any shenanigans for it. Pure equality happened on its own.

*(experiment 4/6 — complementary relationships)*

But there’s another possibility too. What if relationships transfer responsibilities instead of strengths?

Emma fell in love with a bluey who did really well in his job. But this time, instead of learning from his strengths, she covers for his weaknesses. She is now responsible for everything that her partner isn’t doing

Having men and women specialise into different roles is terrible for our society, especially when men already have the power.

Sorry, did I say men? I meant blueys

*[show male female matching]*

I tried simulating it by pairing up random people, and then having them act as an inverse influence to each other. Each partner learns traits that the other one is bad at.

This was really bad, we somehow ended up with NO pinkies in Job A

By the way, the thing about relationships is: not everyone needs to be in one to feel its effects. Maybe Emma hates relationships and gender roles. But if all of Emma’s friends are in relationships where their financial ambition is being suppressed, it will naturally also affect her success as well. And this sticks.

complementary relationships: **F tier**.

In fact, when you remove other influences, The system you get is so unstable, that you can create a patriarchy even when I started from complete equality

I started a new simulation with complete equality between pinkies and blueys. I gave them random partners, and encouraged negative influences. What happens is if you wait long enough, eventually Job A becomes either 100 percent pinkies or 100 percent blueys. Equality is simply not stable at all.

**[TEXT ON SCREEN — TIER LIST SO FAR]**

```
S tier:  Positive relationships
F tier:  Gender quotas (with backlash) | You Go Girl (F-F bias)
D tier:  Gender quotas (with reservation combo)
```
*[More experiments still to come.]*

---

## PART 10 — FIXED FRIENDSHIPS

*(experiment 5/6 — fixed similarity friends)*

Sometimes the best solutions are the ones you never expect

Lets change the way we form friendships.

Lets give every person 4 permanent friends at the start. These will be the friends they are most similar to. And thats it.

No more friendships. No changing of groups. Only fixed friends with very high similarity

When I first tried to simulate this, I honestly thought this would be a disaster. I mean, one would expect that echo chambers form when people never change their friendships right? And in this case, how could an echo chamber not form? Your starting friends will have a similarity score of more than 95 percent.

Also remember that at the start of the simulation, we did not assign gender randomly. blueys are more similar to other blueys, and pinkies to pinkies. This means we rarely see cross gender friendships.

And yet, when I simulated this, things improved — but not enough. After 600 iterations, about a third of Job A is pinkie. Progress, not parity.

How?

*(visual: show a clique of 4 people, and then we show a bridge between 2 cliques, then interconnected cliques)*

Let's look more closely at our friend network. Yes, like minded people form friendships, which leads to the formation of groups. But there are a few people that land between groups. Sometimes, the people most similar to you, may not be very similar to each other. These connections form the bridges that bring our world together.

Previously the moment 2 groups drifted apart, these bridges dissolved. This time, they hold on forever. And through these bridges, influence travels.

Let's look at Emma again

She starts out with childhood friends that are mostly pinkies and very similar to her. but they have very different innate attributes. Eventually, even though they end up becoming completely different from each other, they still remain friends. And this makes all the difference

Maybe the best way to have diversity in our world isn’t to force diverse people together, Maybe all we need is the open mindedness to stay with someone even when they become different.

In this case, permanent bridges helped — but similarity-based friends alone don't close the gap completely.

And you know what, we can do better. Your childhood friends aren’t always the ones most similar to you. They are probably the ones who lived closest to you as a kid, or went to the same school.

If we can make our neighbourhoods more diverse, we can make our friendships diverse too.

So lets do one last experiment

*(experiment 6/6 — diverse childhood friends)*

Permanent, diverse bonds, starting from your childhood. After that, it doesn’t really matter.

You can keep forming new friendships. Moving jobs. Getting into relationships, even bad ones, whatever. As long as you are part of 1 diverse group, thats enough to protect you from echo chambers. And this creates an equality that lasts forever

By iteration 200, we have reached complete equality. And this equality will last forever

**[TEXT ON SCREEN]** `Iteration 50 — Women in Job A: 28%`

**[TEXT ON SCREEN]** `Iteration 200 — Women in Job A: 49%`

By the end of the simulation, people form friendships with 20 other people on average. But it's the 4 diverse friends from their childhood that made all the difference.

**[TEXT ON SCREEN — FINAL TIER LIST]**

```
S tier:  Positive relationships | Diverse childhood friends
B tier:  Fixed similarity friends (partial — ~34% in Job A)
D tier:  Gender quotas (with reservation combo)
F tier:  Gender quotas (with backlash) | You Go Girl | Complementary relationships
```

---

## PART 11 — SOCIAL MEDIA VS ALGORITHMS

I previously said that social media causes radicalization. But that’s not entirely true.

Modern algorithms push content using similarity groups. However, in the earliest days of social media, people stayed connected through permanent links - like mailing lists, or just being facebook friends.

These permanent links are extremely good at keeping people connected, even when they drift apart. It is the best way for you to get exposed to new opinions. You won’t get diverse content from a reel while doomscrolling. You are more likely to get diverse content through your friends' stories. And that’s what social media is good at.

So if you like this video, you get added to a similarity group that will keep pushing more such videos from other creators. But if you subscribe to my channel, you will keep getting my content, even if it's not related to this video at all.

---

## PART 12 — PROBLEMS WITH THE SIMULATION

Before you take any of this too seriously — remember this.

This simulation doesn’t capture the real world completely. Hiring isn’t meritocratic. Discrimination exists. Harassment exists. Institutional barriers we haven’t even talked about. Plus people are filled with a lot of hate. I didn’t put that hate in my world.

And yet even in this sanitised, generous version of the problem — where everything is fair except the starting distribution — the inequality is very tough to fight.

So are men and women equal in the real world? Probably not.

But here is what a real experiment on this question would require. You would need to raise over a 1000 children in a completely controlled environment with controlled influences, and monitor them for more than 30 years.

That experiment has never been run. It almost certainly never will be.

And that's why I chose to simulate it instead.

---

## PART 13 — IMAGINING A BETTER WORLD

And the reason I simulated a world that is almost perfect, is because I like to believe thats what we are capable of. And that the people in my world are capable of even more

- Embracing diversity
- Maintaining friendships (childhood friendships)
- Forming positive relationships

These are the things that make the world a better place.

You might say it will never happen.

**[VISUAL]** Slow zoom out on the graph. All curves visible together.

But look at how much the world has changed in the last fifty years. We see the world around us, and we make the mistake of thinking - this is how it's always been, so this is how it will be

But aren't we already much better than our parents? Why can’t our children be better than us?

If you stayed so far, I applaud you

Because, you would rather imagine something that fails than fail to imagine anything at all.

---

## APPENDIX

### Simulation data referenced in script

| Scenario | Job A female % (iter 600) | Female pay % of male (iter 600) |
|---|---|---|
| Fixed Similarity Friends | 33% single run / 34% mean 20 runs | 68% / 73% |
| Dynamic Friends | 27% / 21% | 47% / 43% |
| Dynamic + F-F Bias | 20% / 11% | 40% / 35% |
| Dynamic + Reservation | 27% / 20% | 47% / 43% |
| Fixed Random Friends | 60% single run / 52% mean 20 runs | 103% / 105% |

### Character reference

| Name | Person # | Innate rank | Starting job | Innate traits |
|---|---|---|---|---|
| Emma | 91 | #1 of 100 | Job C | 70.8, 76.8, 57.7 |
| Rachel | 54 | #12 of 100 | Job B | 49.7, 71.3, 69.9 |

*Emma has the single highest innate job-trait score in the entire population. She starts in Job C. Rachel reaches Job B before losing her network.*

### Colour palette

```
Background:    #0D0D0D
Male nodes:    #4A90D9
Female nodes:  #E8705A
Job A ring:    #FFD700
Job B ring:    #C0C0C0
Job C ring:    #555555
Bridge links:  #F5C842
Text primary:  #FFFFFF
Text muted:    #AAAAAA
```

### Animation notes

- **Node size:** proportional to current job-trait score sum
- **Node border:** gold = Job A, silver = Job B, dim grey = Job C
- **Friendship lines:** thin white 20% opacity (default), amber 60% (bridges), permanent amber (random childhood)
- **Iteration counter:** always visible, bottom or top corner
- **Timer:** top right, monospace, white, counts down during intro only
- **Trait bars:** radial fan from node (overview) or horizontal panel (close-up)
- **Innate layer:** faint inner glow, only briefly visible when mentioned
- **Pacing:** ~0.05s per iteration normal, ~0.5s slow-motion for key moments, ~0.01s fast-forward for long sweeps

