# Session 1: Weekly Catch-Up and DMC Experiments

**Date:** 2026-03-22
**Presenter:** Yad Konrad
**Video:** [YouTube](https://www.youtube.com/live/L3PamTTQFGk)
**Length:** ~60 minutes

## About

Walkthrough of using Claude Code with SutroYaro to:

- Sync Telegram, Google Docs, and GitHub into a weekly catch-up summary
- Run DMC baseline sweep across all methods
- Launch parallel agents for independent experiments
- Create and manage GitHub issues from research findings
- Prepare a presentation report for Meeting #10

## Chapters

| Time | Topic |
|------|-------|
| 0:00 | Intro |
| 0:28 | What is the Sutro Group and the research problem |
| 2:01 | Energy efficiency for AI training, sparse parity as toy problem |
| 3:37 | Showing the repo |
| 4:03 | How I got involved: coding agents as research agents |
| 5:07 | The SutroYaro repository and website |
| 6:56 | What this session covers: Claude Code workflow, not the math |
| 7:52 | Setting up: Claude Code, off-peak hours, voice-to-text |
| 8:16 | Asking the agent to sync and create a weekly catch-up |
| 9:28 | Voice-to-text tools (Handy Computer) |
| 10:05 | Why verbose/thinking mode matters |
| 11:03 | How Claude Code navigates vs RAG-based agents |
| 12:55 | Skills: extending the agent with reusable guidelines |
| 13:16 | Syncing Google Docs and Telegram automatically |
| 15:03 | Reading the weekly catch-up together |
| 17:43 | New metric: DMC (Data Movement Complexity) |
| 19:01 | The CLAUDE.md file: mapping the environment for the agent |
| 20:16 | Writing context vs writing rules for agents |
| 21:35 | Using external models (GLM5) with Claude Code |
| 23:07 | Reviewing the catch-up: public domain, GitHub issues |
| 24:06 | Flipper Zero as a controller for voice-to-text |
| 26:06 | GitHub CLI as an agent tool |
| 27:00 | Expanding the agent scratchpad |
| 28:53 | MkDocs, digital gardens, Obsidian |
| 30:01 | Creating GitHub issues, inventory of needs |
| 31:22 | Erdos problems: using agents for open math research |
| 34:07 | Dispatching parallel agents |
| 35:05 | Avoiding agent babysitting: skip-permissions |
| 36:35 | Long-horizon tasks vs one-at-a-time prompting |
| 37:04 | Running agents in loops, plugins |
| 38:13 | My plugin setup: superpowers, calendar, Tavily |
| 40:51 | Launching parallel agents for DMC experiments |
| 44:57 | Don't trust agent results at face value |
| 46:02 | Reviewing agent outputs, changelog |
| 48:34 | Flipper Zero Mac Remote setup |
| 49:04 | Status line: session time, rate limits, context |
| 49:55 | Creating a branch and PR with verification |
| 51:03 | Team of agents vs sub-agents |
| 53:00 | DMC optimization results |
| 53:50 | Managing multiple agent sessions like an inbox |
| 55:01 | Why I work on research problems with agents |
| 56:16 | Anti-slop skill, posting to Telegram |
| 59:07 | Wrap-up |

## Related

- [Weekly Catch-Up (Mar 16-22)](../catchups/2026-03-22.md)
- [DMC Optimization Findings](../findings/exp_dmc_optimize.md)
- [Meeting #10 Report](../catchups/meeting-10-report.md)

## Transcript

Transcribed with whisper-large-v3 via MLX (Apple Silicon). Timestamps from word-level alignment.

??? note "Full transcript (click to expand)"

    **[00:04]** All right, fingers crossed, this is going to work. I don't know, every time I log into YouTube and OBS, it just slowly gets more complex. But hey, how are you? In case you end up being able to see this later and this all works out magically, I'm going to assume so. So I will start yapping about what is this stream about. Okay, so first of all, welcome. This is going to be one of my first catch-ups about the Sutro group updates.

    **[00:37]** If you don't know what it is, it's a research group currently led by Yaroslav YB. If you know, you know. If you don't know, you're going to be able to find the links. There is a Google. So the group meets weekly in person. And there is a Google Doc that there are a bunch of Google Docs and there's a Telegram group that contains pretty much all the ongoing conversations. What I'm doing is that I'm sort of catching up remotely,

    **[01:11]** not being in person through the Google Docs and the Telegram chat and all the other explorations that I'm doing. To kind of give you context of what the research group currently is sort of focusing on is this idea of finding ways to do AI training, in more in a energy efficient way. To be specific, let's see. I don't know if being specific is actually going to help me

    **[01:41]** make this better. But one of the core questions right now is, I would say trying to figure out if you could train a neural network in like whatever, like some number X, significantly less energy than what it takes to train, I would say, some number X, significantly less energy than what it takes to train, is that a model today. So you can think of it that way. So about energy efficiency for AI training, but also specifically what would it look like if you had a solver or a neural net. Right now, as a toy problem,

    **[02:11]** one of the problems Yaroslav put out there is doing this sparse parity or sparse sum. Obviously that you can think of it as like, that sounds very simple. Can you not make a neural net solve that? Yes, you can. An analogy I think he used, from what I have gathered in the Google docs, it's like how biologists use food flies. So this is a toy problem that's complex enough that would expose the shortcomings of different learners or methods that you're going to use.

    **[02:43]** So for example, if you use a neural net, like for example, FGD, it takes about 120 milliseconds, sounds pretty reasonable, but then you end up using algebraic solver, it takes about 500 milliseconds. So things

    **[03:25]** assume that's the gaussian uh elimination one uh takes about 500 microseconds significantly less yeah so that's the idea okay now i'm going to switch to the other screen if this only works nope that did not work hilarious all right it this works i this is very raw i'm just i'm just

    **[03:56]** assuming this is all going to work out okay so where do i come in into the conversation so once i got the news of uh yarrow and a few other people in the group working on this i thought it would be interesting to try to introduce coding agents to function as research agents to help with gathering more information running experiments and then pretty much creating like this almost rl type of environment but not really where you are

    **[04:29]** creating set of benchmarks you're trying to figure out different trajectories i'm using a mixed language between what the coding agent literature is using and rl so neither group of people would understand that but i'm going to show you how to do that in a minute so let's get started but the idea is to build up a structured code base or rather structured workspace that allows the coding agent to explore it and run experiments that's really in brief that's how i would sort of describe it and in order to make to make things really straightforward

    **[05:07]** we have a repository that contains all of this and it's known as sutro yarrow that is what you see in front of you this is the repository as you see we have a few contributors here and this as of yesterday is for it's a public domain so that way people can easily adopt and contribute to it and uh there's a website okay so this is where it

    **[05:40]** gets interesting there is an ongoing project called the also a telegram group and so i use the coding agent to build up the structured workspace to catch up and just sort of learn of what's going on i try to contribute and this site that you see in front of you there's mk dogs static document is not manage by me this is almost like i were scratch bad between myself and the other

    **[06:13]** contributors that the coding agent uses to um obviously the coding agent directed by us uses to note down for what's going on so everything that i just said if it did not make sense at all which i assume so you can find more about it here but let's say that you are actually a little bit too tired i'm not going to say lazy what you can do is that you can get cloned the repo locally and then you can pull up your own your very own favorite coding agent whatever that would be

    **[06:47]** in my case is going to be cloud code and you can basically start working like so this is what this session is going to be about it's going to be a lot less about the problem set but more about how i use very specifically cloud code and why cloud code and how it works and how it helps to keep this working and then you can pull up your own your very own favorite coding agent whatever that sort of going okay i'm gonna see how we are doing on time the problem is i actually don't have a

    **[07:19]** timer do i have a timer i don't think i do okay seven minutes not bad i'm going to try to keep it under 40 minutes but okay so let me let me see what i have for myself in my outline what you're not seeing is that there are four screens here there is a screen here there is a screen there there is a screen there there's a screen here and each one of them is responsible for reporting different set of agents and this is the one that i'm only showing here okay so i think like the

    **[07:52]** first thing i would like to walk you through is i have this i've caught code open um today is sunday we are past peak hours thanks to the anthropic team we get 2x usage so we are going to be able to make benefit of our cloud account during off peak hours it's like a thing going on right now so the first thing i will do is i will just try to ask the agent to catch me up to date for what has been going on and so

    **[08:24]** you're going to see something that looks like a little bit of magic which is a voice to text thing uh so i'm going to enable that now hey can you sync i'm going to bring the can you sync the the google docs and also can you check the telegram groups specifically the chat yaroslav chat yad and then the other ones for new activities new ideas and then once you have the data locally

    **[08:59]** let's create a weekly catch-up for the week before we start exploring and running experiments but there are a few others the one that uh i would recommend if you don't want to spend

    **[09:34]** any penny on is handy computer i have a customized version of this which makes me think that i need to contribute it back but this kind of they all somewhat do the same which allows you to take the voice and then transcribe it and do some punctuation and then paste it into the selected code and then you can do the same thing with the selected input in this case the selected input will be the terminal okay so i have verbose modon so if you have claude you might initially you're

    **[10:05]** not going to be able to see this sort of thinking process i have verbose on which is really important i do recommend you to enable it the reason that is is because specifically cloud code their coding agent people call harness there are different ways whatever the term whatever makes you feel good about understanding what this is i don't know what the term is but i think it's a good one like i'm gonna go with agent their agent functions a bit differently than most other coding agents so for example if you have used let's say cursor or anti-gravity or any of these ide levels

    **[10:39]** one of the ways they try to do exploration is by creating a retrieval or rather like a rack type of base for the code base or for the structured workspace that i have built that's terrible and the reason is terrible because you constantly you constantly need to update it and it just takes up extra two three gigabyte of space however the way cloud code does it it pretty much functions like a navigator meaning that it has a loose awareness of what's in the

    **[11:10]** structure workspace and then start aggressively doing a combination of grep and then the tools are available so if you were to look at the actual behavior over overall the agent is just sort of doing a bunch of like scratchback tasks perform the tasks scratchback task perform the tasks and these tasks are oftentimes are bash call like cli level command call and oftentimes happen to be the

    **[11:41]** set of guidelines that you have set up for the agent and so if i scroll all the way back you will see right from the get-go when i was yapping earlier what the agent starts doing it starts invoking this sutro think skill and sutro context and you're like what is that my computer doesn't have that and you would be right the reason that is is because um early on i was doing a lot of these uh things manually meaning that i was basically telling the agent what to write and what to do

    **[12:17]** and so eventually what we ended up doing we ended up getting the agent to write a skill that contains the guideline and so so this is one way that you can expand the coding agent beyond repeatedly every time to go like do this don't do that do this but no this is important but no that's important because you often forget and it seems actually really tedious thing to do instead you create this almost like a um almost like a navigation tool i keep using the word

    **[12:54]** navigation in the description box just to give you the description box just to give you the because that's like the only really good analogy that I have right now. The agent being almost like a self-driving car and you are like the operator, but then the self-driving car needs like some like needs a LIDAR. It needs like a bunch of sensors. And then so you sort of provide those sensors. I don't know if this is really clicking, but that's really what's going on. So the first skill is synchronizing Google Doc. That's good. The second skill is running this Telegram script that is in the code base.

    **[13:28]** So if you were to pull this, you can create your own Telegram token. And then your agent locally will be able to, in a similar way, synchronize the context from the Telegram channel. And then it starts running. So for example, this is an improvement we can do. Right away, it seems like it is doing a Python. I think this Python loop could potentially be a toolkit that we can write. It looks like I am.

    **[14:02]** This is me complaining about the video not being uploaded in 6K. I know. All right. So this is the catch up. I'm going to go to the bottom. So as you can see, it's reading up on the last Yaroslav knowledge sprint. So these are the notes that Yaro have mentioned. Whenever you see this sort of weird dark dark color, that is basically what the thinking trace of the agent.

    **[14:32]** This is like the internal reasoning model, given the information that has and what it sort of needs to do. So if you recall, I was saying that the way the agent functions, it creates like almost like a scratch pad and then performs tasks on that scratch pad. This is part of that scratch pad. And in order to be a better, better driver of the agent, you wouldn't want to see what it actually is reading. And you could basically stop it from sometimes doing catastrophic mistakes.

    **[15:03]** Okay, so this is my catch up. So let's read it together. I actually have not been able to catch up. We can catch up. 16 860 messages. I want to make sure this is readable on the screen because if not, we are in trouble. No, I think this is readable. We'll see. Either this is going to be terrible or barely understandable, but that's good enough. That's a good enough start for us. Okay, so what is going on? So there is a new metric.

    **[15:34]** Yaroslav presented the roadmap key outcomes, data movement, data movement complexity, Ding et al. The new homework is to optimize DMC instead of ARD. So ARD is this. Um, so if we actually said the agent created a word glossary for us somewhere. I need to find it, but I don't think so.

    **[16:04]** This is the ARD baseline. It explains what it is and why it's a baseline. Um, I don't want to say what the acronym stands for. Out of blue, but I also, what is ARD? Help.

    **[16:40]** I'm partially playing stupid here that I want you to see what it would look like if you were to do this by yourself and you felt like you were getting stuck. Um, but I actually want to find, oh, there we go. Nope, that is not it. That is not it at all. Average for use distance. So this was introduced when at the beginning,

    **[17:12]** I think there were the problems set in the neural net were presented by 20 bits and only three of them. we're a nonzero. I believe so. I might be wrong about a lot of these things because I've read them and then I end up going on my own tangent. But just so, so just so you know, this is what it looks like for me when I catch up. Oftentimes I remember like, oh, that's what it's there for. And, uh, so it looks like that there is a bit of a change there.

    **[17:43]** Meta goal iterate on the process of going from metric plus problem specification to fast sequence. Experiments, not just solving the problem, but making the solving fast. I like that idea. Meeting video posted in YouTube. Oh, it looks like the videos in YouTube. I didn't know that there's an actual notes. We don't, we don't have think yet. That's strange. Okay. It looks like there might be visitors outside researchers stopping by.

    **[18:19]** So, and then you also have, I believe he introduced like this idea yesterday that I briefly responded back to. Um, and then, and so this is interesting part. You noted that. So this is, so, uh, somewhere in the, somewhere in the cloud MD file. Uh, if you, if we look at it, there is this idea of the agent being able to identify whose computer it's on based on the username and the gist. Uh, and so that's why it's actually able to say that you noted that our 33 experiments and by use referring to me.

    **[18:58]** Yeah. Uh, who is mentioned here. So if you're curious to know what's going on, there is a bit of description in one of the earlier videos, but overall most coding agents, um, require like this almost map file. This people call context file, whatever. The best way to, uh, think of it, what the cloud MD file does. It is a default way for the agent to load information about the environment it's in.

    **[19:32]** So whatever you will put in here will be a part of the agents sort of steps and tasks. Normally people love to put in rules in here. I don't think rules work because eventually, the agent will, because if you put a rule in here, like don't delete this, don't run that command. The agent is like, yes, you know, copy that. But then if you're about a hundred thousand tokens in that rule gets buried,

    **[20:02]** the agent will eventually override that. I'm pretty sure they're the team behind the coding agents is a whole field of research trying to fix that. But one of the better ways to avoid that is to just describe what is the environment? What are the goals? What are some scripts that are risky? Uh, what are the things that you need to run rather than do's and don'ts? So anyway, so that's what this file is. This file is really just getting the agent to let's look at it in a nicer way. So it is a map in a sense. It tries to introduce the other files that are here,

    **[20:35]** what their purposes are and where the agent should write and read from talks about some of the core concepts. It talks about some of the current methods that are introduced. So, this was my favorite finding using the SMT backtracking. That's a whole different story. We'll get to that. But yeah, so, so this is like sort of the essential part of the actual agent.

    **[21:06]** So if you end up using Codex, for example, or open code, it will be able to actually load the same information. You would hope so. If it doesn't, then that's where you, you kind of need to restructure or create a branch that fits how your agent would be able to better understand. But if we were to use cloud code for now, then you would be safe. Here's an interesting part. So cloud code by itself,

    **[21:37]** the agent, the harness can actually work with external models. So for example, I use GLM5, which is a, I'm going to actually just test that out just to demonstrate to you what I'm trying to talk about. But I'm going to open up in a different session because I don't want you to see my, my keys. That would be a problem. That would be a big problem if I accidentally exposed my key.

    **[22:09]** But so check this out. So I just opened cloud code. It's using GLM5. GLM5 is, one of the more recent models from ZAI that is quote unquote, SOTA state of the art for some of the benchmarks around software engineering and human eval bench. How is this relevant? It is relevant because the agent is like a core part of doing all the exploration and then you can change the models.

    **[22:44]** And so part of the meta problem we're solving could actually be, this, at least in my, in my perspective is like, oh, how can I get this thing to be able to run for in a certain shape or form for a longer period or approaches and things like that. So, okay. So let's see. Let's see what sort of, so I guess like this is sort of the catch up. Yaroslav asked if Sutro can be designed a public domain. I said, yes. Yaroslav is going to do Manhattan.

    **[23:15]** That's super cool. Okay. So I created a few, GitHub issues. What is due tomorrow? Get agents to improve the sparsperity using DMC, not ARB. So that's a good task. Iterate on prompts and meta approaches. Okay. So here's what I'm going to do. I'm going to, okay. So if you're, if you're seeing this contraption, this is just my flipper zero. And I have created a controller and the controller,

    **[23:47]** have about, six, my action space is six, which is getting the, getting the voice activated, stop and click. And so that's what I'm going to use during going forward to avoid doing a lot of clicky clack on there. Also, I broke my tab button, but that's a different story. Okay. So can you create a new section that's called weekly catch up? And we put this summary as is, as you wrote it for me in there. And,

    **[24:17]** and include everything, include the suggested free experiment plan, include the due tomorrow, include the date, include the summaries and whatnot. And ideally we also finish our Google doc synchronization for the last meeting. So we have full context. And then also after that, we will need to create a task list for the homework. And also a task list for the suggested pre-experiment plan.

    **[24:58]** So I just hit that and then I hit the middle one. And so the way I, this, this program is open source if you wanted to access it later, but you just put it on your flip zero and I currently don't have, or the, the microphone extension is not really, really so fit. So that's why I'm using my headphone, my noise counseling headphone. And so the mic is pretty close and it works. And so, yeah, so that's what's going on. Okay. So it is going to perform those tasks.

    **[25:31]** As you can see, it's synchronizing. So with, so this is the important part. This is how you want, this is how you take a coding agent. It becomes research agent without me explicitly saying that we have a Google doc script where you should use it to synchronize. It knows that because somewhere the way we have described how the lab works and the workspace works, all that information now is embedded within the coding agent. So that's just food for thought thinking about what are some of the other things that you can do with the agent.

    **[26:05]** Okay. So while that's happening, I am going to go to the GitHub repo. I'm going to look at, I think there was a PR. Now the PR is good. Let's see. These are all me here. Well, what I wanted to do is I actually wanted to see if we can get the transcript for the, for the video from the last session.

    **[26:40]** Okay. So it looks like listed a TLDR. Now, let me make everything. Weekly catch up page. Okay. So that's what the agent is doing right now. Okay. So I think one of the, let me see how many minutes are we in? We're about 26 minutes in. Okay, not bad. So I went through the weekly sync. Oh, okay. What I need to talk about, what I actually wanted to talk about is that the way you can expand the skill set of the agent,

    **[27:14]** besides like the doc and all that type of stuff is getting the agent to understand the local command line tools that you have. And so one of those command line tools that I have is the GitHub CLI, which can read and write to get help. And I have like this granular. So I have the token that can only write to a specific repo to avoid any crazy thing, which doesn't really happen unless you explicitly tell the agent to do.

    **[27:47]** But I have it dedicated to only this repo in this scenario. So all the stuff you see here is not me. If you saw my GitHub issues, questionable writing, but so that's, so that's how you can expand the agents. Scratchpad. I'm going to carefully say scratchpad. I'm not going to say memory because I feel like, so, some of these words are overused and it's just used for everything. So it's not really a memory.

    **[28:18]** It's not a recall. It doesn't automatically knows it exists there. You have to explicitly tell the agent. Hey, do you remember we wrote down our task list somewhere? And then the agent goes like, oh, wait. Yes. I see that in my lab document somewhere. You have described that we can expand the memory or rather, where I store data to get up and places like that. So, so that's that.

    **[28:48]** Okay. So let's see what's going on. Also, I read somewhere that MK docs is being deprecated. So it might be a good idea for us to think about a new way to present this information. I didn't want to introduce a lot of like new ones, but I have actually created like a little, uh, so I have created like a little project. Oh, by the way, there is, this is a blog that describes the suture yarrow or specifically the suture group research problems that I understand in a bit of depth.

    **[29:25]** But this is my version of MK docs. It connects to obsidian and it's just a set of markdowns, but it connects all together and it is like a digital garden. Digital garden is a terminology used by the law history behind it, but it's used by static website generator hipsters. If I can put it that way. Okay. So,

    **[29:55]** okay. So the agent created the catch up and did that get up issues. Okay. So I'm going to, okay. So did we create the issues on GitHub for the tasks for us to follow through? And while you're going to create those issues, are there any other information that we need to update as in, since we're moving on to trying the DMC instead of the ARD,

    **[30:26]** let's try to take inventory of what we have and what are some other baseline fundamental toolkits that we need to create. And then after that, the GitHub issues, or maybe all of these can be GitHub issues. Okay. So I, one thing that I wanted to point out is that when I started out the first week, I was not even using GitHub because I tried to start out really simple when I build up these structured workspaces.

    **[30:56]** Also for context, this is my third project that is around similar, like this idea of you bring in a bunch of documents and you try to build up, or you try to turn the cloud code into a research agent. So if you're familiar with, I need to get the pronunciation right. I think it's Erdos. I don't think it's Erdos. But anyway, Erdos problems are a bunch of open-ended math problems to be specific around 1183.

    **[31:32]** Earlier this year, a few folks started being able to solve them. Just using chat GPT and the codex, I believe, or the GPT-5, one of the models. And so this gave me an idea. What would it look like if we created a workspace locally and created enough context and you created like a lab like type of thing that the coding agent would run experiments without doing without cheating. This is really important because one of the things you don't want the agent to do is plagiarizing results coming up with,

    **[32:08]** theorem, like proofs that exist out there or rather not unique or new. And so that was like my first experiment. And it was interesting because I found a few things out, which is what would make the coding agent tick. What are like some external information you can use? How can you verify? I think verification is like a really important part. And so I ended up creating like this whole lean,

    **[32:38]** runtime module that every time it comes up with a partial program, it tries to run it to verify it. And more importantly, to reduce the error rate, I created a skill that explains what would be a good way to actually write the formalized version of the lean, of the, of the thing. The theorem. Or whatever it tried. It tries to actually generate.

    **[33:10]** And so this actually reduced not only error rate, but I was able to reproduce some of those problems with a not a unique solution, but not 100% matching what the other teams had found. I don't know. How am I if I'm saying right like it came up with a solution, but it's not comparable to the solution of the problems that were on. There. So in a way, you can actually reproduce them without cheating. And so that was like,

    **[33:40]** that's super cool. So the agent can actually solve open in the problems. If you just set up enough path for the agent to run like these experiments. So that was like one of them. Okay, so I'm going to go back here. Okay, so the agent is still the agent is still so the agent is creating these GitHub issues. Yeah, yeah, yeah, so let's go. Let's see what's going on here.

    **[34:13]** No wrong repository. Okay, so now we have 16 issues. Look at that. Look at that at tracker integration too fast agent notification bridge. Oh, so I was trying to figure out a way to basically get the agent to post back to telegram so we can create a chat group that will report everybody's experiments. If if like 10 people were running this on their own,

    **[34:45]** what would that look like? But yeah. So let's let's okay. Maybe let's talk about something that would be relevant, which is how do you get the agent to actually run in loops because a lot of people one of the things. That they encounter when they first these coding agents again, and I say I'm very very specific cloud code because that's the one where I do all my benchmarks on they have to babysit the agent in the terminal meaning that it's like oh this one.

    **[35:19]** Okay hit next one of the ways that you can solve this issue is if you recall I'm going to exit out when I started out. I basically had this dangerously skip permission the dangerously skip permission. As I've said before is a bit misleading because the permissions it is skipping is not always dangerous. Oftentimes you might actually end up having the reverse effect, which is you are saying accept edit accept edit or yes continue.

    **[35:54]** Yes continue and just because you're so bored and tired doing inter inter you'll enter to something that's actually you don't want the agent to do. So you would rather give the agent more autonomy. In a sense that it can run more steps. But what you want to do on your end is coming up with better mapping or rather the the context for the agent what to do. So you kind of add to the harness rather than trying to pull it apart.

    **[36:29]** I don't know if that makes sense. But the way I do this is by as you seeing in front of me instead of doing it. One task at a time. I try to describe this sort of long running Horizon task for the agent and be like, all right, do these 10 things or plan these 10 things and I want you to run them. So that's one way. That's one way. You can expand the agent to do things besides sort of hey do this do that. I'm trying to see how many issues they created.

    **[37:00]** The other way is that you can explicitly have the agent run. Run in loops. So there are a couple. So there are a couple of projects to do that. One of the ones that the Anthropix team. So if you go to cloud code skills, you will be able to find the official skills are recommended here. Very specifically. You will find one that's called.

    **[37:30]** Oh. Oh, I guess like, it is not in here anymore. Well, that's so there used to be one that's called Rolf Wiggum, which was created by the indie dev. Oh my bad. That is a plug-in. I think I think it might be in the plug-in section. I might be wrong. I don't know. This thing's moving pretty fast. There are things that get deprecated. But anyway, long story short,

    **[38:01]** there are plugins that you can use. You can install that will make the agent behave in a different way. And so if I were to show you my plug-in set, you will be able to see I have the agent SDK. I have the the ceiling LSP server. I have the code review feature dev. Some of these are disabled as you can see plug-in dev. The Ralph Wiggum security guidance superpowers.

    **[38:33]** Apple calendar, draw IO, Mac OS Automator, Tavili. And so that's only the plugins to this project. Globally. I have other plugins that are not showing up here. So what are plugins? Plugins for cloud code is a combination between these guideline documentation. That's currently called skills scripts that the agent can run. And I forgot what the other thing is. It's a combination of like three things basically.

    **[39:06]** And and the coding agent cloud code have an API that allows it to either sort of like run like hook to it. It's called hooks. It's almost like a I guess like if you're using I'm trying to figure out like if you're a web dev, it's almost like a web hook. Maybe you can think of it that way. Like as in that when the task performs at a zero cost,

    **[39:38]** Claude is going to be able to run that script for you. So one example of the hook is actually the notification system. So you're not able to hear it. But when the task ends, I hear a notification in my ear and that's a hook. So that's one of the examples. You can get the coding agent build up like more complex environments or environment Navigator. Okay. I feel like I'm repeating myself, but that is really a part of it. A part of it is like just hearing these things of seeing what's going on.

    **[40:10]** And then eventually being able to have you or yourself try to either poke around at this or creating a version of this yourself. So I think like that's what I somewhat also recommend. This is only one version of creating this environment. There are actually many ways to go about creating this structure workspace. Because at the end of the day, the main goal is actually getting the agent from the experiments. Okay, so created eight issues this and that good news harness.

    **[40:43]** Already confused DNC for all five core. We can run the baseline. Okay, so this is like a little fun thing that I'm going to do. One of the ways besides those scripts in the last three weeks, the anthropic team have or the cloud code team. I've introduced. This thing called team of agents, which is your main agent actually being able to Spown like spawn and run. It spawns like a bunch of agents and then and then you will be able to perform all of these tasks in parallel.

    **[41:21]** And that's what I'm going to do right now. Okay, so it seems like some of these tasks can be done in parallel. Would it make sense to deploy? A set of agents to be able to do all of them and report back. So let's see. Sometimes the tasks might be dependent, but I think so far it makes sense to me.

    **[41:55]** The user wants to paralyze the work across multiple agents. Let me think about which tasks are true. So things like things

    **[42:28]** So this is one of the plugins that I have installed. It contains a bunch of skill sets. So, okay, so this is what is in here. You see it contains hooks, which is a bunch of, could be scripts, but in this case, it is just a file. Agents is a category, so you can actually define agents. Okay, let's see what's going on.

    **[43:04]** Okay, so I have disabled. So I have, I guess I could have enabled that previously. When you run the multi-agent, there's a plugin that you can install, and what it does, it actually splits the screen for you, and you will see all the, you will be able to observe what the other agents are. But right now, it is actually in here. So that's what these are. And you can tap into each terminal, and you can send a message specifically to that agent.

    **[43:35]** But we're not going to do that. We are going to try to review what the work is and what is going on. But yeah, so we have three set of agents that are right now in progress. Okay, it will let us know. Sauteed for one minute and 16 seconds. Gotta love the vocab. All right.

    **[44:06]** So where were we? We're talking about superpowers. So yeah, superpowers, pretty interesting. But I think of a lot of these plugins is really just a scaffold for you to be able to build on top of. And so I have a fork of this, and I have a fork of this. I have added my own set of quote unquote superpowers, which means that it makes the agent to be able to perform tasks that normally might not. Normally, you have to define it from scratch.

    **[44:38]** You're like, well, I want you to do it this way. And some of those tasks you don't even know, like don't even know how to, let's say, secure a Linux server. So that could be a good sort of skill to have the agent to be able to follow. Okay. Two of three agents are done. The DMC baseline results are in and they are significant. Okay.

    **[45:08]** I'm going to be very honest. I don't really buy into this. So that's the other skillset that you need to acquire, which is not believing what the agent says at face value. Because probably. So one of the agents that I've been working with, the error that I noticed in our workspace was that it was not making it clear to the agent that when I asked to run an experiment, it needs to run it on a GPU, especially those that are GPU related. So that took a week for me to realize.

    **[45:41]** And because of that, some of the reports you can see had inflated numbers, but I was able to catch up on that by having the agent review the task. So the task is done. So change log is always a really good way to keep track of really what's going on and what the agent is doing and whatnot. So let's see, this is running sparse parity fast. Let's see what's going on.

    **[46:12]** All right. All three agents delivered. 14 runs across three challenges disagree on the best. Best method cam wins on ARD and GF two wins on DMC GF two. I'm trying to next steps now unblocked. These were waiting on the baseline sweeps. Let me to kick off the next wave of agent.

    **[46:48]** Yes. If the next set of agents are going to generate independent outputs without overriding the work we have done, since we want to be able to keep track of the experiments one after another and then compare and then verify. Also the program that I have it on the flipper zero is actually. So, okay. So what's going on is that it tries to, it did a bunch of tasks, but then.

    **[47:19]** So again, I actually don't agree. This could have been paralyzed as well. And. And. I'm running parallel, but it didn't run them in parallel, but that's fine. The, the program that I'm running on here is, Ooh, you guys are seeing all my projects. Is called flipper zero. Oh, there we go. Flipper zero applications. So you can install this on your flipper zero and what it does.

    **[47:50]** Mac remote. So you can, it will connect to Bluetooth and then. So down. So I use option, option slash. That's what toggles my voice to text up is escape because in cloth code or most of the coding agents, that's how you interrupt. And then left is arrow up, up and down. So, so I'm using a display up and down.

    **[48:21]** So I'm using a display up and down to try to navigate the options that oftentimes the agent gives you and then enter. But then also I have two actions dedicated to. Okay. So that's pretty good. And then I have exit and then I have switched mode, but that's all right. I just got a notification saying it ended. Look at that. You see, that's a good use of the hook, getting cloth code to send you a notification. Okay. So there's no overlaps with two dispatch.

    **[48:52]** And then it has now three agents running in the terminal. You might see this thing here that might not look like how yours look like. And what's going on here is that this is called a status line. It is a way to show information about the ongoing session. So this has been running. Have I been 40 minutes? Wow. 49 minutes. Okay. I've been yapping for 49 minutes. So this is this is the time that I've been running.

    **[49:26]** This is five hour up to my cool down, which I have not had my limit. I try to avoid hitting my limits, but that's a reminder for my limit. This is my get diff. We have added 4000 lines, removed 51. And then this is my context. And then this is the model. And then that's the branch. I forgot I should have created a branch and I should have not made a commit. Well, I might actually do that right now.

    **[49:59]** Can you create a new branch and commit all our work under a branch called Yad? So that way we follow the protocol that we have set for everyone else. For contributions. And then ideally we create a PR and then through the PR, we're going to have almost like a release in there.

    **[50:33]** But also before we do the commit, it's really important to do first verify. Second, make sure we are going to include all the updates. And the change log. And third, we need a report that explains the results for the experiments that we have run. So we can present it or somebody to be able to present it tomorrow. So the cool thing is that when when the agents run in the background, your main loop opens up so you can actually continue interacting with the main agent while you have these sort of like sub agents or the parallel agents.

    **[51:17]** If you're interested in understanding how the Cloud Code's team of agents versus sub agents work, it's actually defined here. So I have like everybody else who spends a little bit too much on the coding. Like everybody tried to sort of create a hacky version of this. I have a hacky version of it. This one is actually really cool. That team of agents. And the reason that is, is just because it's a little bit more complex. Just because hold on.

    **[51:49]** Let me zoom in. I don't really know. Okay. So they could have a shared task, which I think is really important. So the difference is that trying to think about. So the difference is that oftentimes here in the work tree, you might end up having overlap. One thing you want to avoid is the agent trying to edit the same file. And then you have like you could have like this. Almost like infinite error loop.

    **[52:19]** But here the agents are actually aware of the work stream going on. And so that's really cool because you can actually now think about if you recall earlier how I was describing instead of babysitting the agent trying to think about the long term, the long run task. If you knew that's how the agents behave, you can actually think about like, oh, maybe I need to come up with like a master plan and then have the team of agents. Implement that.

    **[52:50]** Okay. I'm going to run this for the sake of time, trying to keep it under an hour. I said 45 minutes went to an hour. So the MC optimization number 22 new best method found. Okay. We have a bit of a flickering problem in my terminal. I will explain what's going on later. But I have a problem. I have modified my item. In fact, I'm working on creating a terminal emulator that's specific for the coding agent.

    **[53:25]** But because this this is like really annoying, I can't read it. As you can see, what's happening is that every time it updates it, it scrolls all the way to up. But I'm going to try to scroll up to read what it will us. This is frustrating. All right. I'm going to let it finish. But yeah, so I have changed my tabs to vertical.

    **[53:56]** The way I try to think about it is that it's almost like my inbox. And I'm like trying to it's like almost like our C channel, whatever Yahoo, whatever messenger tool you use. And every time I send a message in one session that gets priority. And then when I have my other agents, they sort of like. Deprioritize. And then when that one gets an update, it goes back to the top of the stack for me to follow up. But.

    **[54:27]** Okay. Pretty frustrating. That's all right. I'm going to wait until it ends. But yeah, that's pretty much how much time I try to sort of spend around an hour, hour and a half for these different sort of research exploratory problems. I have a few, as I mentioned, other problems that I've been dealing with. I kind of explore in similar way. By no measure. I've I've I know how I think I've read about this. I know how to read a proof and maybe you're right.

    **[54:59]** But it would take me like three days. But then using an agent to actually help you and work with you on that is very cool. And so that's like why I got involved in some of the other heavier math problems. And then I think for here, I'm pretty interested. Because. Yeah. For a bunch of reasons. I'm obviously I think this is like a fundamental research type of problem. First reason. Second of all, is that usually around research groups that takes an orthodox approaches, you end up learning things that you would normally not learn.

    **[55:36]** And so I think this is somewhat unorthodox. So it's making a mistake. Let's see. It says Yaroslav. Yeah. So they have things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they

    **[56:16]** I think that's a bug somewhere. It's fine, but it's all we. Okay, so these are the final touches. We are 56 minutes in. And when these are published, you will be able to find it in, I need to close, way too many tabs are open.

    **[56:48]** You will be able to find it here. This is where Cybertron.GitHub.IO. There, shall I proceed with the branch creation? Yes, as far as we have verified the results before committing and the report makes sense. Also make sure that you're going to use the anti-slop scale. To avoid the sloppy AI language in the report.

    **[57:25]** So I created a scale that tries to basically remove these common vocab and terminology that you often see in reports or like long form of text of agents. And then you can also use the The only reason that is, is that, I mean, obviously we all know we're using agents. The only problem is that it oftentimes actually makes it really vague.

    **[57:55]** Like not because of X, because of Y. Oh, like, are you just using that just because you don't have a better reference. So anyway, but that's what some of the skills are. You know, it's really useful. So let's see. The key issue is fine. Look at that. It did. It did. It did write garbage. It did. Sometimes it does write garbage. So I'm glad we caught it. It's getting to correct it. And it usually does become a lot more readable

    **[58:26]** when you try to remove that language. And then one of the other things I'm going to ask the agent to do is create a summary set of messages to post to the Telegram. And. So things like things

    **[58:57]** I don't know why it changed a bunch of headers. Oh, it changed it because it was in that week's report. Okay, okay, fair, fair. All right, I think that this is going to be one hour. I'm going to try to keep it one hour. I think we're going to go over a bit one hour, but that's pretty much it for this week. I will share the findings and you will be able to see the report. I hope this was helpful. I hope this was useful by any means.

    **[59:29]** If not, feel free to ask away questions.

