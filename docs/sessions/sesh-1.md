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

    [00:04] All right, fingers crossed, this is going to work.
    [00:08] I don't know, every time I log into YouTube and OBS,
    [00:11] it just slowly gets more complex.
    [00:14] But hey, how are you?
    [00:16] In case you end up being able to see this later
    [00:18] and this all works out magically,
    [00:21] I'm going to assume so.
    [00:23] So I will start yapping about what is this stream about.
    [00:28] Okay, so first of all, welcome.
    [00:30] This is going to be one of my first catch-ups
    [00:34] about the Sutro group updates.
    [00:37] If you don't know what it is,
    [00:39] it's a research group currently led by Yaroslav YB.
    [00:47] If you know, you know.
    [00:48] If you don't know, you're going to be able to find the links.
    [00:51] There is a Google.
    [00:52] So the group meets weekly in person.
    [00:58] And there is a Google Doc that there are a bunch of Google Docs
    [01:02] and there's a Telegram group that contains
    [01:04] pretty much all the ongoing conversations.
    [01:06] What I'm doing is that I'm sort of catching up remotely,
    [01:11] not being in person through the Google Docs
    [01:13] and the Telegram chat
    [01:14] and all the other explorations that I'm doing.
    [01:17] To kind of give you context
    [01:19] of what the research group currently is sort of focusing on
    [01:22] is this idea of finding ways to do AI training,
    [01:28] in more in a energy efficient way.
    [01:31] To be specific, let's see.
    [01:37] I don't know if being specific is actually going to help me
    [01:41] make this better.
    [01:42] But one of the core questions right now is,
    [01:45] I would say trying to figure out
    [01:47] if you could train a neural network in like whatever,
    [01:51] like some number X,
    [01:54] significantly less energy than what it takes to train,
    [01:58] I would say, some number X, significantly less energy than what it takes to train,
    [01:58] is that a model today.
    [01:59] So you can think of it that way.
    [02:01] So about energy efficiency for AI training,
    [02:05] but also specifically what would it look like
    [02:07] if you had a solver or a neural net.
    [02:10] Right now, as a toy problem,
    [02:11] one of the problems Yaroslav put out there
    [02:13] is doing this sparse parity or sparse sum.
    [02:19] Obviously that you can think of it as like,
    [02:21] that sounds very simple.
    [02:22] Can you not make a neural net solve that?
    [02:24] Yes, you can.
    [02:26] An analogy I think he used,
    [02:28] from what I have gathered in the Google docs,
    [02:30] it's like how biologists use food flies.
    [02:32] So this is a toy problem that's complex enough
    [02:35] that would expose the shortcomings of different learners
    [02:40] or methods that you're going to use.
    [02:43] So for example, if you use a neural net,
    [02:46] like for example, FGD, it takes about 120 milliseconds,
    [02:50] sounds pretty reasonable,
    [02:51] but then you end up using algebraic solver,
    [02:54] it takes about 500 milliseconds.
    [03:25] assume that's the gaussian uh elimination one uh takes about 500 microseconds significantly less
    [03:37] yeah so that's the idea okay now i'm going to switch to the other screen if this only works
    [03:44] nope that did not work hilarious all right it this works i this is very raw i'm just i'm just
    [03:56] assuming this is all going to work out okay so where do i come in into the conversation so once i
    [04:03] got the news of uh yarrow and a few other people in the group working on this i thought it would
    [04:10] be interesting to try to introduce coding agents
    [04:14] to function as research agents to help with gathering more information running experiments
    [04:21] and then pretty much creating like this almost rl type of environment but not really where you are
    [04:29] creating set of benchmarks you're trying to figure out different trajectories i'm using a mixed
    [04:37] language between what the coding agent literature is using and rl so neither group of people would
    [04:43] understand that but i'm going to show you how to do that in a minute so let's get started
    [04:44] but the idea is to build up a structured code base or rather structured workspace
    [04:51] that allows the coding agent to explore it and run experiments that's really in brief
    [04:58] that's how i would sort of describe it and in order to make to make things really straightforward
    [05:07] we have a repository that contains all of this and it's known as
    [05:15] sutro yarrow that is what you see in front of you this is the repository
    [05:21] as you see we have a few contributors here and this as of yesterday is for it's a public domain so that
    [05:33] way people can easily adopt and contribute to it and uh there's a website okay so this is where it
    [05:40] gets interesting there is an ongoing project called the
    [06:02] also a telegram group and so i use the coding agent to build up the structured workspace to catch up and just sort of learn of what's going on i try to contribute and this site that you see in front of you there's mk dogs static document is not manage by me this is almost like i were scratch bad between myself and the other
    [06:13] contributors that the coding agent uses to um obviously the coding agent directed by us uses to
    [06:22] note down for what's going on so everything that i just said if it did not make sense at all which
    [06:27] i assume so you can find more about it here but let's say that you are actually a little bit too
    [06:33] tired i'm not going to say lazy what you can do is that you can get cloned the repo locally and
    [06:40] then you can pull up your own your very own favorite coding agent whatever that would be
    [06:47] in my case is going to be cloud code and you can basically start working like so this is what this
    [06:56] session is going to be about it's going to be a lot less about the problem set but more about how
    [07:02] i use very specifically cloud code and why cloud code and how it works and how it helps to keep
    [07:10] this working and then you can pull up your own your very own favorite coding agent whatever that
    [07:10] sort of going okay i'm gonna see how we are doing on time the problem is i actually don't have a
    [07:19] timer do i have a timer i don't think i do okay seven minutes not bad i'm going to try to keep
    [07:26] it under 40 minutes but okay so let me let me see what i have for myself in my outline what you're
    [07:35] not seeing is that there are four screens here there is a screen here there is a screen there
    [07:40] there is a screen there there's a screen here and each one of them is responsible for reporting
    [07:46] different set of agents and this is the one that i'm only showing here okay so i think like the
    [07:52] first thing i would like to walk you through is i have this i've caught code open um today is sunday
    [07:59] we are past peak hours thanks to the anthropic team we get 2x usage so we are going to be able to
    [08:08] make benefit of our
    [08:10] cloud account during off peak hours it's like a thing going on right now so the first thing i will
    [08:16] do is i will just try to ask the agent to catch me up to date for what has been going on and so
    [08:24] you're going to see something that looks like a little bit of magic which is a voice to text
    [08:31] thing uh so i'm going to enable that now hey can you sync i'm going to bring the can you sync the
    [08:40] the google docs and also can you check the telegram groups specifically the chat yaroslav
    [08:51] chat yad and then the other ones for new activities new ideas and then once you have the data locally
    [08:59] let's create a weekly catch-up for the week before we start exploring and running experiments
    [09:28] but there are a few others the one that uh i would recommend if you don't want to spend
    [09:34] any penny on is handy computer i have a customized version of this which makes me think that i need
    [09:41] to contribute it back but this kind of they all somewhat do the same which allows you to take the
    [09:49] voice and then transcribe it and do some punctuation and then paste it into the selected
    [09:56] code and then you can do the same thing with the selected input in this case the selected input
    [09:58] will be the terminal okay so i have verbose modon so if you have claude you might initially you're
    [10:05] not going to be able to see this sort of thinking process i have verbose on which is really important
    [10:11] i do recommend you to enable it the reason that is is because specifically cloud code their coding
    [10:18] agent people call harness there are different ways whatever the term whatever makes you feel good
    [10:24] about understanding what this is i don't know what the term is but i think it's a good one
    [10:26] like i'm gonna go with agent their agent functions a bit differently than most other coding agents so
    [10:32] for example if you have used let's say cursor or anti-gravity or any of these ide levels
    [10:39] one of the ways they try to do exploration is by creating a retrieval or rather like a rack
    [10:45] type of base for the code base or for the structured workspace that i have built
    [10:50] that's terrible and the reason is terrible because you constantly you constantly need to update
    [10:56] it and it just takes up extra two three gigabyte of space however the way cloud code does it
    [11:03] it pretty much functions like a navigator meaning that it has a loose awareness of what's in the
    [11:10] structure workspace and then start aggressively doing a combination of grep and then the tools
    [11:16] are available so if you were to look at the actual behavior over overall the agent is just sort of
    [11:25] doing a bunch of like scratchback tasks perform the tasks scratchback task perform the tasks and
    [11:32] these tasks are oftentimes are bash call like cli level command call and oftentimes happen to be the
    [11:41] set of guidelines that you have set up for the agent and so if i scroll all the way back you
    [11:47] will see right from the get-go when i was yapping earlier what the agent starts doing it starts
    [11:53] invoking this
    [11:55] sutro think skill and sutro context and you're like what is that my computer doesn't have that
    [12:03] and you would be right the reason that is is because um early on i was doing a lot of these
    [12:10] uh things manually meaning that i was basically telling the agent what to write and what to do
    [12:17] and so eventually what we ended up doing we ended up getting the agent to write a skill
    [12:25] that contains the guideline and so so this is one way that you can expand the coding agent beyond
    [12:33] repeatedly every time to go like do this don't do that do this but no this is important but
    [12:40] no that's important because you often forget and it seems actually really tedious thing to do
    [12:46] instead you create this almost like a um almost like a navigation tool i keep using the word
    [12:54] navigation in the description box just to give you the description box just to give you the
    [12:55] because that's like the only really good analogy that I have right now.
    [12:59] The agent being almost like a self-driving car and you are like the operator,
    [13:04] but then the self-driving car needs like some like needs a LIDAR.
    [13:08] It needs like a bunch of sensors.
    [13:10] And then so you sort of provide those sensors.
    [13:12] I don't know if this is really clicking, but that's really what's going on.
    [13:16] So the first skill is synchronizing Google Doc.
    [13:21] That's good.
    [13:22] The second skill is running this Telegram script that is in the code base.
    [13:28] So if you were to pull this, you can create your own Telegram token.
    [13:33] And then your agent locally will be able to, in a similar way,
    [13:38] synchronize the context from the Telegram channel.
    [13:41] And then it starts running.
    [13:44] So for example, this is an improvement we can do.
    [13:49] Right away, it seems like it is doing a Python.
    [13:52] I think this Python loop could potentially be a toolkit that we can write.
    [13:58] It looks like I am.
    [14:02] This is me complaining about the video not being uploaded in 6K.
    [14:06] I know.
    [14:08] All right.
    [14:08] So this is the catch up.
    [14:10] I'm going to go to the bottom.
    [14:12] So as you can see, it's reading up on the last Yaroslav knowledge sprint.
    [14:18] So these are the notes that Yaro have mentioned.
    [14:23] Whenever you see this sort of weird dark dark color,
    [14:27] that is basically what the thinking trace of the agent.
    [14:32] This is like the internal reasoning model,
    [14:36] given the information that has and what it sort of needs to do.
    [14:40] So if you recall, I was saying that the way the agent functions,
    [14:42] it creates like almost like a scratch pad and then performs tasks on that scratch pad.
    [14:47] This is part of that scratch pad.
    [14:50] And in order to be a better,
    [14:52] better driver of the agent,
    [14:54] you wouldn't want to see what it actually is reading.
    [14:58] And you could basically stop it from sometimes doing catastrophic mistakes.
    [15:03] Okay, so this is my catch up.
    [15:04] So let's read it together.
    [15:05] I actually have not been able to catch up.
    [15:08] We can catch up.
    [15:12] 16 860 messages.
    [15:14] I want to make sure this is readable on the screen because if not,
    [15:17] we are in trouble.
    [15:18] No, I think this is readable.
    [15:21] We'll see.
    [15:22] Either this is going to be terrible or barely understandable,
    [15:26] but that's good enough.
    [15:27] That's a good enough start for us.
    [15:29] Okay, so what is going on?
    [15:31] So there is a new metric.
    [15:34] Yaroslav presented the roadmap key outcomes,
    [15:37] data movement,
    [15:37] data movement complexity,
    [15:39] Ding et al.
    [15:40] The new homework is to optimize DMC instead of ARD.
    [15:45] So ARD is this.
    [15:52] so if we actually said the agent created a word glossary for us somewhere.
    [15:58] I need to find it,
    [16:01] but I don't think so.
    [16:04] This is the ARD baseline.
    [16:06] It explains what it is and why it's a baseline.
    [16:13] I don't want to say what the acronym stands for.
    [16:19] Out of blue,
    [16:20] but I also,
    [16:27] what is ARD?
    [16:33] Help.
    [16:40] I'm partially playing stupid here that I want you to see what it would look like
    [16:46] if you were to do this by yourself and you felt like you were getting stuck.
    [16:51] but I actually want to find,
    [16:53] there we go.
    [16:54] Nope,
    [16:55] that is not it.
    [16:57] That is not it at all.
    [17:04] Average for use distance.
    [17:07] So this was introduced when at the beginning,
    [17:12] I think there were the problems set in the neural net were presented by 20 bits and only three of them.
    [17:22] we're a nonzero.
    [17:23] I believe so.
    [17:26] I might be wrong about a lot of these things because I've read them and then I end up going on my own tangent.
    [17:31] But just so,
    [17:32] so just so you know,
    [17:33] this is what it looks like for me when I catch up.
    [17:36] Oftentimes I remember like,
    [17:37] that's what it's there for.
    [17:38] And,
    [17:39] so it looks like that there is a bit of a change there.
    [17:43] Meta goal iterate on the process of going from metric plus problem specification to fast sequence.
    [17:49] Experiments,
    [17:51] not just solving the problem,
    [17:53] but making the solving fast.
    [17:55] I like that idea.
    [17:56] Meeting video posted in YouTube.
    [17:59] it looks like the videos in YouTube.
    [18:00] I didn't know that there's an actual notes.
    [18:03] We don't,
    [18:05] we don't have think yet.
    [18:07] That's strange.
    [18:09] Okay.
    [18:12] It looks like there might be visitors outside researchers stopping by.
    [18:20] and then you also have,
    [18:21] I believe he introduced like this idea yesterday that I briefly responded back to.
    [18:28] and then,
    [18:29] and so this is interesting part.
    [18:31] You noted that.
    [18:32] So this is,
    [18:34] somewhere in the,
    [18:36] somewhere in the cloud MD file.
    [18:40] if you,
    [18:40] if we look at it,
    [18:41] there is this idea of the agent being able to identify whose computer it's on based on the username and the gist.
    [18:49] and so that's why it's actually able to say that you noted that our 33 experiments and by use referring to me.
    [18:58] Yeah.
    [18:59] who is mentioned here.
    [19:01] So if you're curious to know what's going on,
    [19:04] there is a bit of description in one of the earlier videos,
    [19:07] but overall most coding agents,
    [19:11] require like this almost map file.
    [19:14] This people call context file,
    [19:18] whatever.
    [19:19] The best way to,
    [19:21] think of it,
    [19:23] what the cloud MD file does.
    [19:25] It is a default way for the agent to load information about the environment it's in.
    [19:32] So whatever you will put in here will be a part of the agents sort of steps and tasks.
    [19:41] Normally people love to put in rules in here.
    [19:44] I don't think rules work because eventually,
    [19:49] the agent will,
    [19:50] because if you put a rule in here,
    [19:51] like don't delete this,
    [19:52] don't run that command.
    [19:54] The agent is like,
    [19:55] yes,
    [19:55] you know,
    [19:56] copy that.
    [19:57] But then if you're about a hundred thousand tokens in that rule gets buried,
    [20:02] the agent will eventually override that.
    [20:05] I'm pretty sure they're the team behind the coding agents is a whole field of research trying to fix that.
    [20:11] But one of the better ways to avoid that is to just describe what is the environment?
    [20:16] What are the goals?
    [20:17] What are some scripts that are risky?
    [20:19] what are the things that you need to run rather than do's and don'ts?
    [20:21] So anyway,
    [20:22] so that's what this file is.
    [20:23] This file is really just getting the agent to let's look at it in a nicer way.
    [20:29] So it is a map in a sense.
    [20:31] It tries to introduce the other files that are here,
    [20:35] what their purposes are and where the agent should write and read from talks about some of the core concepts.
    [20:45] It talks about some of the current methods that are introduced.
    [20:49] this was my favorite finding using the SMT backtracking.
    [20:55] That's a whole different story.
    [20:57] We'll get to that.
    [20:58] But yeah,
    [20:59] so this is like sort of the essential part of the actual agent.
    [21:06] So if you end up using Codex,
    [21:08] for example,
    [21:09] or open code,
    [21:10] it will be able to actually load the same information.
    [21:15] You would hope so.
    [21:16] If it doesn't,
    [21:17] then that's where you,
    [21:19] you kind of need to restructure or create a branch that fits how your agent would be able to better understand.
    [21:26] But if we were to use cloud code for now,
    [21:30] then you would be safe.
    [21:33] Here's an interesting part.
    [21:35] So cloud code by itself,
    [21:37] the agent,
    [21:37] the harness can actually work with external models.
    [21:41] So for example,
    [21:43] I use GLM5,
    [21:46] which is a,
    [21:49] I'm going to actually just test that out just to demonstrate to you what I'm trying to talk about.
    [21:54] But I'm going to open up in a different session because I don't want you to see my,
    [22:02] my keys.
    [22:03] That would be a problem.
    [22:05] That would be a big problem if I accidentally exposed my key.
    [22:09] But so check this out.
    [22:12] So I just opened cloud code.
    [22:15] It's using GLM5.
    [22:17] GLM5 is,
    [22:19] one of the more recent models from ZAI that is quote unquote,
    [22:26] SOTA state of the art for some of the benchmarks around software engineering and human eval bench.
    [22:35] How is this relevant?
    [22:36] It is relevant because the agent is like a core part of doing all the exploration and then you can change the models.
    [22:44] And so part of the meta problem we're solving could actually be,
    [22:49] this,
    [22:49] at least in my,
    [22:51] in my perspective is like,
    [22:52] how can I get this thing to be able to run for in a certain shape or form for a longer period or approaches and things like that.
    [23:02] okay.
    [23:03] So let's see.
    [23:04] Let's see what sort of,
    [23:05] so I guess like this is sort of the catch up.
    [23:07] Yaroslav asked if Sutro can be designed a public domain.
    [23:10] I said,
    [23:11] yes.
    [23:13] Yaroslav is going to do Manhattan.
    [23:15] That's super cool.
    [23:17] Okay.
    [23:17] So I created a few,
    [23:19] GitHub issues.
    [23:20] What is due tomorrow?
    [23:22] Get agents to improve the sparsperity using DMC,
    [23:25] not ARB.
    [23:26] So that's a good task.
    [23:27] Iterate on prompts and meta approaches.
    [23:30] Okay.
    [23:31] So here's what I'm going to do.
    [23:32] I'm going to,
    [23:34] okay.
    [23:36] So if you're,
    [23:36] if you're seeing this contraption,
    [23:38] this is just my flipper zero.
    [23:40] And I have created a controller and the controller,
    [23:47] have about,
    [23:49] six,
    [23:49] my action space is six,
    [23:51] which is getting the,
    [23:52] getting the voice activated,
    [23:54] stop and click.
    [23:56] And so that's what I'm going to use during going forward to avoid doing a lot of clicky clack on there.
    [24:02] Also,
    [24:03] I broke my tab button,
    [24:04] but that's a different story.
    [24:06] Okay.
    [24:06] So can you create a new section that's called weekly catch up?
    [24:10] And we put this summary as is,
    [24:14] as you wrote it for me in there.
    [24:17] And,
    [24:17] and include everything,
    [24:19] include the suggested free experiment plan,
    [24:22] include the due tomorrow,
    [24:25] include the date,
    [24:26] include the summaries and whatnot.
    [24:30] And ideally we also finish our Google doc synchronization for the last meeting.
    [24:38] So we have full context.
    [24:40] And then also after that,
    [24:43] we will need to create a task list for the homework.
    [24:47] And also a task list for the suggested pre-experiment plan.
    [24:58] So I just hit that and then I hit the middle one.
    [25:02] And so the way I,
    [25:04] this,
    [25:05] this program is open source if you wanted to access it later,
    [25:09] but you just put it on your flip zero and I currently don't have,
    [25:14] or the,
    [25:14] the microphone extension is not really,
    [25:17] really so fit.
    [25:18] So that's why I'm using my headphone,
    [25:20] my noise counseling headphone.
    [25:22] And so the mic is pretty close and it works.
    [25:24] And so,
    [25:24] yeah,
    [25:24] so that's what's going on.
    [25:26] Okay.
    [25:28] So it is going to perform those tasks.
    [25:31] As you can see,
    [25:32] it's synchronizing.
    [25:33] So with,
    [25:34] so this is the important part.
    [25:36] This is how you want,
    [25:37] this is how you take a coding agent.
    [25:38] It becomes research agent without me explicitly saying that we have a Google doc script where you should use it to synchronize.
    [25:47] It knows that because somewhere the way we have described how the lab works and the workspace works,
    [25:54] all that information now is embedded within the coding agent.
    [25:58] So that's just food for thought thinking about what are some of the other things that you can do with the agent.
    [26:05] Okay.
    [26:06] So while that's happening,
    [26:07] I am going to go to the GitHub repo.
    [26:12] I'm going to look at,
    [26:13] I think there was a PR.
    [26:14] Now the PR is good.
    [26:18] Let's see.
    [26:20] These are all me here.
    [26:25] Well,
    [26:26] what I wanted to do is I actually wanted to see if we can get the transcript for the,
    [26:34] for the video from the last session.
    [26:40] Okay.
    [26:41] So it looks like listed a TLDR.
    [26:43] Now,
    [26:44] let me make everything.
    [26:44] Weekly catch up page.
    [26:47] Okay.
    [26:47] So that's what the agent is doing right now.
    [26:51] Okay.
    [26:51] So I think one of the,
    [26:54] let me see how many minutes are we in?
    [26:58] We're about 26 minutes in.
    [27:00] Okay,
    [27:00] not bad.
    [27:02] So I went through the weekly sync.
    [27:06] okay.
    [27:07] What I need to talk about,
    [27:08] what I actually wanted to talk about is that the way you can expand the skill set of the agent,
    [27:14] besides like the doc and all that type of stuff is getting the agent to understand the local command line tools that you have.
    [27:25] And so one of those command line tools that I have is the GitHub CLI,
    [27:29] which can read and write to get help.
    [27:33] And I have like this granular.
    [27:37] So I have the token that can only write to a specific repo to avoid any crazy thing,
    [27:43] which doesn't really happen unless you explicitly tell the agent to do.
    [27:47] But I have it dedicated to only this repo in this scenario.
    [27:51] So all the stuff you see here is not me.
    [27:55] If you saw my GitHub issues,
    [27:58] questionable writing,
    [27:59] but so that's,
    [28:00] so that's how you can expand the agents.
    [28:06] Scratchpad.
    [28:07] I'm going to carefully say scratchpad.
    [28:09] I'm not going to say memory because I feel like,
    [28:13] some of these words are overused and it's just used for everything.
    [28:16] So it's not really a memory.
    [28:18] It's not a recall.
    [28:19] It doesn't automatically knows it exists there.
    [28:24] You have to explicitly tell the agent.
    [28:26] Hey, do you remember we wrote down our task list somewhere?
    [28:30] And then the agent goes like,
    [28:31] oh, wait.
    [28:31] Yes.
    [28:32] I see that in my lab document somewhere.
    [28:36] You have described that we can expand the memory or rather,
    [28:42] where I store data to get up and places like that.
    [28:47] so that's that.
    [28:48] Okay.
    [28:49] So let's see what's going on.
    [28:53] Also,
    [28:53] I read somewhere that MK docs is being deprecated.
    [28:57] So it might be a good idea for us to think about a new way to present this information.
    [29:04] I didn't want to introduce a lot of like new ones,
    [29:07] but I have actually created like a little,
    [29:10] so I have created like a little project.
    [29:14] by the way,
    [29:15] there is,
    [29:15] this is a blog that describes the suture yarrow or specifically the suture group research problems that I understand in a bit of depth.
    [29:25] But this is my version of MK docs.
    [29:28] It connects to obsidian and it's just a set of markdowns,
    [29:35] but it connects all together and it is like a digital garden.
    [29:41] Digital garden is a terminology used by the law history behind it,
    [29:48] but it's used by static website generator hipsters.
    [29:53] If I can put it that way.
    [29:54] Okay.
    [29:55] okay.
    [29:56] So the agent created the catch up and did that get up issues.
    [30:00] Okay.
    [30:01] So I'm going to,
    [30:03] okay.
    [30:03] So did we create the issues on GitHub for the tasks for us to follow through?
    [30:09] And while you're going to create those issues,
    [30:14] are there any other information that we need to update as in,
    [30:18] since we're moving on to trying the DMC instead of the ARD,
    [30:26] let's try to take inventory of what we have and what are some other baseline fundamental toolkits that we need to create.
    [30:33] And then after that,
    [30:35] the GitHub issues,
    [30:35] or maybe all of these can be GitHub issues.
    [30:38] Okay.
    [30:45] So I,
    [30:46] one thing that I wanted to point out is that when I started out the first week,
    [30:50] I was not even using GitHub because I tried to start out really simple when I build up these structured workspaces.
    [30:56] Also for context,
    [30:58] this is my third project that is around similar,
    [31:02] like this idea of you bring in a bunch of documents and you try to build up,
    [31:08] or you try to turn the cloud code into a research agent.
    [31:13] So if you're familiar with,
    [31:19] I need to get the pronunciation right.
    [31:22] I think it's Erdos.
    [31:23] I don't think it's Erdos.
    [31:25] But anyway,
    [31:25] Erdos problems are a bunch of open-ended math problems to be specific around 1183.
    [31:32] Earlier this year,
    [31:34] a few folks started being able to solve them.
    [31:38] Just using chat GPT and the codex,
    [31:42] I believe,
    [31:42] or the GPT-5,
    [31:45] one of the models.
    [31:47] And so this gave me an idea.
    [31:48] What would it look like if we created a workspace locally and created enough context and you created like a lab like type of thing that the coding agent would run experiments without doing without cheating.
    [32:01] This is really important because one of the things you don't want the agent to do is plagiarizing results coming up with,
    [32:08] theorem,
    [32:09] like proofs that exist out there or rather not unique or new.
    [32:14] And so that was like my first experiment.
    [32:16] And it was interesting because I found a few things out,
    [32:20] which is what would make the coding agent tick.
    [32:24] What are like some external information you can use?
    [32:27] How can you verify?
    [32:29] I think verification is like a really important part.
    [32:31] And so I ended up creating like this whole lean,
    [32:38] runtime module that every time it comes up with a partial program,
    [32:46] it tries to run it to verify it.
    [32:48] And more importantly,
    [32:51] to reduce the error rate,
    [32:54] I created a skill that explains what would be a good way to actually write the formalized version of the lean,
    [33:05] of the,
    [33:06] of the thing.
    [33:06] The theorem.
    [33:07] Or whatever it tried.
    [33:08] It tries to actually generate.
    [33:10] And so this actually reduced not only error rate,
    [33:13] but I was able to reproduce some of those problems with a not a unique solution,
    [33:20] but not 100% matching what the other teams had found.
    [33:27] I don't know.
    [33:28] How am I if I'm saying right like it came up with a solution,
    [33:31] but it's not comparable to the solution of the problems that were on.
    [33:36] There.
    [33:36] So in a way,
    [33:37] you can actually reproduce them without cheating.
    [33:40] And so that was like,
    [33:40] that's super cool.
    [33:41] So the agent can actually solve open in the problems.
    [33:45] If you just set up enough path for the agent to run like these experiments.
    [33:52] So that was like one of them.
    [33:53] Okay,
    [33:54] so I'm going to go back here.
    [33:55] Okay,
    [33:55] so the agent is still the agent is still so the agent is creating these GitHub issues.
    [34:04] Yeah,
    [34:05] yeah,
    [34:05] yeah,
    [34:07] so let's go.
    [34:09] Let's see what's going on here.
    [34:13] No wrong repository.
    [34:17] Okay,
    [34:18] so now we have 16 issues.
    [34:19] Look at that.
    [34:22] Look at that at tracker integration too fast agent notification bridge.
    [34:29] so I was trying to figure out a way to basically get the agent to post back to
    [34:35] telegram so we can create a chat group that will report everybody's experiments.
    [34:42] If if like 10 people were running this on their own,
    [34:45] what would that look like?
    [34:48] But yeah.
    [34:54] So let's let's okay.
    [34:55] Maybe let's talk about something that would be relevant,
    [34:58] which is how do you get the agent to actually run in loops because a lot of people one of the things.
    [35:05] That they encounter when they first these coding agents again,
    [35:09] and I say I'm very very specific cloud code because that's the one where I do all my benchmarks on they have to
    [35:15] babysit the agent in the terminal meaning that it's like oh this one.
    [35:19] Okay hit next one of the ways that you can solve this issue is if you recall I'm going to exit out when I started
    [35:29] out.
    [35:29] I basically had this dangerously skip permission the dangerously skip permission.
    [35:35] As I've said before is a bit misleading because the permissions it is skipping is not always dangerous.
    [35:45] Oftentimes you might actually end up having the reverse effect,
    [35:49] which is you are saying accept edit accept edit or yes continue.
    [35:54] Yes continue and just because you're so bored and tired doing inter inter you'll enter to something that's actually you don't want the agent to do.
    [36:02] So you would rather give the agent more autonomy.
    [36:05] In a sense that it can run more steps.
    [36:09] But what you want to do on your end is coming up with better mapping or rather the the context for the agent what to do.
    [36:22] So you kind of add to the harness rather than trying to pull it apart.
    [36:29] I don't know if that makes sense.
    [36:30] But the way I do this is by as you seeing in front of me instead of doing it.
    [36:35] One task at a time.
    [36:36] I try to describe this sort of long running Horizon task for the agent and be like,
    [36:42] all right,
    [36:42] do these 10 things or plan these 10 things and I want you to run them.
    [36:47] So that's one way.
    [36:48] That's one way.
    [36:49] You can expand the agent to do things besides sort of hey do this do that.
    [36:58] I'm trying to see how many issues they created.
    [37:00] The other way is that you can explicitly have the agent run.
    [37:04] Run in loops.
    [37:06] So there are a couple.
    [37:08] So there are a couple of projects to do that.
    [37:10] One of the ones that the Anthropix team.
    [37:13] So if you go to cloud code skills,
    [37:16] you will be able to find the official skills are recommended here.
    [37:23] Very specifically.
    [37:25] You will find one that's called.
    [37:33] Oh, I guess like,
    [37:34] it is not in here anymore.
    [37:36] Well,
    [37:40] that's so there used to be one that's called Rolf Wiggum,
    [37:45] which was created by the indie dev.
    [37:47] Oh my bad.
    [37:48] That is a plug-in.
    [37:49] I think I think it might be in the plug-in section.
    [37:52] I might be wrong.
    [37:53] I don't know.
    [37:54] This thing's moving pretty fast.
    [37:57] There are things that get deprecated.
    [37:58] But anyway,
    [37:59] long story short,
    [38:01] there are plugins that you can use.
    [38:04] You can install that will make the agent behave in a different way.
    [38:10] And so if I were to show you my plug-in set,
    [38:13] you will be able to see I have the agent SDK.
    [38:18] I have the the ceiling LSP server.
    [38:22] I have the code review feature dev.
    [38:25] Some of these are disabled as you can see plug-in dev.
    [38:29] The Ralph Wiggum security guidance superpowers.
    [38:33] Apple calendar,
    [38:35] draw IO,
    [38:36] Mac OS Automator,
    [38:39] Tavili.
    [38:40] And so that's only the plugins to this project.
    [38:45] Globally.
    [38:46] I have other plugins that are not showing up here.
    [38:48] So what are plugins?
    [38:50] Plugins for cloud code is a combination between these guideline documentation.
    [38:55] That's currently called skills scripts that the agent can run.
    [39:01] And I forgot what the other thing is.
    [39:03] It's a combination of like three things basically.
    [39:06] And and the coding agent cloud code have an API that allows it to either sort of like run like hook to it.
    [39:15] It's called hooks.
    [39:16] It's almost like a I guess like if you're using I'm trying to figure out like if you're a web dev,
    [39:27] it's almost like a web hook.
    [39:31] Maybe you can think of it that way.
    [39:32] Like as in that when the task performs at a zero cost,
    [39:38] Claude is going to be able to run that script for you.
    [39:41] So one example of the hook is actually the notification system.
    [39:45] So you're not able to hear it.
    [39:47] But when the task ends,
    [39:49] I hear a notification in my ear and that's a hook.
    [39:51] So that's one of the examples.
    [39:53] You can get the coding agent build up like more complex environments or environment Navigator.
    [40:01] Okay.
    [40:02] I feel like I'm repeating myself,
    [40:04] but that is really a part of it.
    [40:07] A part of it is like just hearing these things of seeing what's going on.
    [40:10] And then eventually being able to have you or yourself try to either poke around at this or creating a version of this yourself.
    [40:22] So I think like that's what I somewhat also recommend.
    [40:25] This is only one version of creating this environment.
    [40:27] There are actually many ways to go about creating this structure workspace.
    [40:31] Because at the end of the day,
    [40:33] the main goal is actually getting the agent from the experiments.
    [40:36] Okay, so created eight issues this and that good news harness.
    [40:43] Already confused DNC for all five core.
    [40:45] We can run the baseline.
    [40:47] Okay, so this is like a little fun thing that I'm going to do.
    [40:51] One of the ways besides those scripts in the last three weeks,
    [40:56] the anthropic team have or the cloud code team.
    [41:00] I've introduced.
    [41:00] This thing called team of agents,
    [41:03] which is your main agent actually being able to Spown like spawn and run.
    [41:11] It spawns like a bunch of agents and then and then you will be able to perform all of these tasks in parallel.
    [41:21] And that's what I'm going to do right now.
    [41:23] Okay, so it seems like some of these tasks can be done in parallel.
    [41:28] Would it make sense to deploy?
    [41:30] A set of agents to be able to do all of them and report back.
    [41:43] So let's see.
    [41:45] Sometimes the tasks might be dependent,
    [41:47] but I think so far it makes sense to me.
    [41:55] The user wants to paralyze the work across multiple agents.
    [41:58] Let me think about which tasks are true.
    [42:28] So this is one of the plugins that I have installed.
    [42:35] It contains a bunch of skill sets.
    [42:37] So, okay, so this is what is in here.
    [42:39] You see it contains hooks, which is a bunch of, could be scripts,
    [42:43] but in this case, it is just a file.
    [42:49] Agents is a category, so you can actually define agents.
    [42:57] Okay, let's see what's going on.
    [43:04] Okay, so I have disabled.
    [43:08] So I have, I guess I could have enabled that previously.
    [43:13] When you run the multi-agent, there's a plugin that you can install,
    [43:17] and what it does, it actually splits the screen for you,
    [43:20] and you will see all the, you will be able to observe
    [43:23] what the other agents are.
    [43:24] But right now, it is actually in here.
    [43:28] So that's what these are.
    [43:30] And you can tap into each terminal,
    [43:32] and you can send a message specifically to that agent.
    [43:35] But we're not going to do that.
    [43:37] We are going to try to review what the work is
    [43:43] and what is going on.
    [43:44] But yeah, so we have three set of agents
    [43:46] that are right now in progress.
    [43:54] Okay, it will let us know.
    [43:57] Sauteed for one minute and 16 seconds.
    [44:00] Gotta love the vocab.
    [44:03] All right.
    [44:06] So where were we?
    [44:07] We're talking about superpowers.
    [44:09] So yeah, superpowers, pretty interesting.
    [44:11] But I think of a lot of these plugins is really just a scaffold
    [44:17] for you to be able to build on top of.
    [44:20] And so I have a fork of this,
    [44:22] and I have a fork of this.
    [44:24] I have added my own set of quote unquote superpowers,
    [44:28] which means that it makes the agent
    [44:31] to be able to perform tasks that normally might not.
    [44:35] Normally, you have to define it from scratch.
    [44:38] You're like, well, I want you to do it this way.
    [44:40] And some of those tasks you don't even know,
    [44:43] like don't even know how to, let's say,
    [44:48] secure a Linux server.
    [44:50] So that could be a good sort of skill to have the agent
    [44:54] to be able to follow.
    [44:55] Okay.
    [44:55] Two of three agents are done.
    [44:57] The DMC baseline results are in and they are significant.
    [45:04] Okay.
    [45:08] I'm going to be very honest.
    [45:11] I don't really buy into this.
    [45:13] So that's the other skillset that you need to acquire,
    [45:17] which is not believing what the agent says at face value.
    [45:21] Because probably.
    [45:22] So one of the agents that I've been working with,
    [45:24] the error that I noticed in our workspace was that
    [45:29] it was not making it clear to the agent
    [45:30] that when I asked to run an experiment,
    [45:33] it needs to run it on a GPU,
    [45:35] especially those that are GPU related.
    [45:38] So that took a week for me to realize.
    [45:41] And because of that,
    [45:42] some of the reports you can see had inflated numbers,
    [45:46] but I was able to catch up on that
    [45:49] by having the agent review the task.
    [45:53] So the task is done.
    [45:54] So change log is always a really good way
    [45:56] to keep track of really what's going on
    [45:59] and what the agent is doing and whatnot.
    [46:02] So let's see, this is running sparse parity fast.
    [46:09] Let's see what's going on.
    [46:12] All right.
    [46:14] All three agents delivered.
    [46:17] 14 runs across three challenges disagree on the best.
    [46:24] Best method cam wins on ARD and GF two wins on DMC GF two.
    [46:32] I'm trying to next steps now unblocked.
    [46:38] These were waiting on the baseline sweeps.
    [46:41] Let me to kick off the next wave of agent.
    [46:48] Yes.
    [46:48] If the next set of agents are going to generate
    [46:54] independent outputs without overriding the work we have done,
    [46:57] since we want to be able to keep track of the experiments
    [47:01] one after another and then compare and then verify.
    [47:09] Also the program that I have it on the flipper zero is
    [47:12] actually.
    [47:13] So, okay.
    [47:14] So what's going on is that it tries to,
    [47:16] it did a bunch of tasks, but then.
    [47:19] So again, I actually don't agree.
    [47:21] This could have been paralyzed as well.
    [47:23] And.
    [47:24] And.
    [47:24] I'm running parallel, but it didn't run them in parallel, but that's fine.
    [47:28] The, the program that I'm running on here is, Ooh,
    [47:33] you guys are seeing all my projects.
    [47:37] Is called flipper zero.
    [47:40] Oh, there we go.
    [47:40] Flipper zero applications.
    [47:44] So you can install this on your flipper zero and what it does.
    [47:50] Mac remote.
    [47:53] So you can, it will connect to Bluetooth and then.
    [48:00] So down.
    [48:02] So I use option, option slash.
    [48:04] That's what toggles my voice to text up is escape because in cloth code or
    [48:11] most of the coding agents, that's how you interrupt.
    [48:14] And then left is arrow up, up and down.
    [48:19] So, so I'm using a display up and down.
    [48:21] So I'm using a display up and down to try to navigate the options that
    [48:24] oftentimes the agent gives you and then enter.
    [48:29] But then also I have two actions dedicated to.
    [48:34] Okay.
    [48:34] So that's pretty good.
    [48:36] And then I have exit and then I have switched mode, but that's all right.
    [48:41] I just got a notification saying it ended.
    [48:44] Look at that.
    [48:45] You see, that's a good use of the hook, getting cloth code to send you a notification.
    [48:49] Okay.
    [48:49] So there's no overlaps with two dispatch.
    [48:52] And then it has now three agents running in the terminal.
    [48:57] You might see this thing here that might not look like how yours look like.
    [49:04] And what's going on here is that this is called a status line.
    [49:09] It is a way to show information about the ongoing session.
    [49:14] So this has been running.
    [49:16] Have I been 40 minutes?
    [49:18] Wow.
    [49:19] 49 minutes.
    [49:20] Okay.
    [49:20] I've been yapping for 49 minutes.
    [49:22] So this is this is the time that I've been running.
    [49:26] This is five hour up to my cool down, which I have not had my limit.
    [49:32] I try to avoid hitting my limits, but that's a reminder for my limit.
    [49:37] This is my get diff.
    [49:39] We have added 4000 lines, removed 51.
    [49:42] And then this is my context.
    [49:45] And then this is the model.
    [49:48] And then that's the branch.
    [49:51] I forgot I should have created a branch and I should have not made a commit.
    [49:55] Well, I might actually do that right now.
    [49:59] Can you create a new branch and commit all our work under a branch called Yad?
    [50:12] So that way we follow the protocol that we have set for everyone else.
    [50:19] For contributions.
    [50:21] And then ideally we create a PR and then through the PR, we're going to have almost like a release in there.
    [50:33] But also before we do the commit, it's really important to do first verify.
    [50:44] Second, make sure we are going to include all the updates.
    [50:47] And the change log.
    [50:48] And third, we need a report that explains the results for the experiments that we have run.
    [50:56] So we can present it or somebody to be able to present it tomorrow.
    [51:03] So the cool thing is that when when the agents run in the background, your main loop opens up so you can actually continue interacting with the main agent while you have these sort of like sub agents or the parallel agents.
    [51:17] If you're interested in understanding how the Cloud Code's team of agents versus sub agents work, it's actually defined here.
    [51:31] So I have like everybody else who spends a little bit too much on the coding.
    [51:37] Like everybody tried to sort of create a hacky version of this.
    [51:40] I have a hacky version of it.
    [51:42] This one is actually really cool.
    [51:44] That team of agents.
    [51:45] And the reason that is, is just because it's a little bit more complex.
    [51:47] Just because hold on.
    [51:49] Let me zoom in.
    [51:50] I don't really know.
    [51:51] Okay.
    [51:51] So they could have a shared task, which I think is really important.
    [51:56] So the difference is that trying to think about.
    [52:03] So the difference is that oftentimes here in the work tree, you might end up having overlap.
    [52:10] One thing you want to avoid is the agent trying to edit the same file.
    [52:15] And then you have like you could have like this.
    [52:17] Almost like infinite error loop.
    [52:19] But here the agents are actually aware of the work stream going on.
    [52:24] And so that's really cool because you can actually now think about if you recall earlier how I was describing instead of babysitting the agent trying to think about the long term, the long run task.
    [52:38] If you knew that's how the agents behave, you can actually think about like, oh, maybe I need to come up with like a master plan and then have the team of agents.
    [52:47] Implement that.
    [52:50] Okay.
    [52:51] I'm going to run this for the sake of time, trying to keep it under an hour.
    [52:57] I said 45 minutes went to an hour.
    [53:00] So the MC optimization number 22 new best method found.
    [53:08] Okay.
    [53:08] We have a bit of a flickering problem in my terminal.
    [53:12] I will explain what's going on later.
    [53:15] But I have a problem.
    [53:17] I have modified my item.
    [53:20] In fact, I'm working on creating a terminal emulator that's specific for the coding agent.
    [53:25] But because this this is like really annoying, I can't read it.
    [53:29] As you can see, what's happening is that every time it updates it, it scrolls all the way to up.
    [53:35] But I'm going to try to scroll up to read what it will us.
    [53:44] This is frustrating.
    [53:46] All right.
    [53:47] I'm going to let it finish.
    [53:50] But yeah, so I have changed my tabs to vertical.
    [53:56] The way I try to think about it is that it's almost like my inbox.
    [54:01] And I'm like trying to it's like almost like our C channel, whatever Yahoo, whatever messenger tool you use.
    [54:08] And every time I send a message in one session that gets priority.
    [54:14] And then when I have my other agents, they sort of like.
    [54:19] Deprioritize.
    [54:19] And then when that one gets an update, it goes back to the top of the stack for me to follow up.
    [54:26] But.
    [54:27] Okay.
    [54:28] Pretty frustrating.
    [54:30] That's all right.
    [54:31] I'm going to wait until it ends.
    [54:33] But yeah, that's pretty much how much time I try to sort of spend around an hour, hour and a half for these different sort of research exploratory problems.
    [54:43] I have a few, as I mentioned, other problems that I've been dealing with.
    [54:47] I kind of explore in similar way.
    [54:49] By no measure.
    [54:54] I've I've I know how I think I've read about this.
    [54:57] I know how to read a proof and maybe you're right.
    [54:59] But it would take me like three days.
    [55:01] But then using an agent to actually help you and work with you on that is very cool.
    [55:06] And so that's like why I got involved in some of the other heavier math problems.
    [55:13] And then I think for here, I'm pretty interested.
    [55:16] Because.
    [55:17] Yeah.
    [55:20] For a bunch of reasons.
    [55:22] I'm obviously I think this is like a fundamental research type of problem.
    [55:25] First reason.
    [55:26] Second of all, is that usually around research groups that takes an orthodox approaches, you end up learning things that you would normally not learn.
    [55:36] And so I think this is somewhat unorthodox.
    [55:39] So it's making a mistake.
    [55:42] Let's see.
    [55:45] It says Yaroslav.
    [55:47] Yeah.
    [55:52] So they have things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they've done things they
    [56:16] I think that's a bug somewhere.
    [56:18] It's fine, but it's all we.
    [56:24] Okay, so these are the final touches.
    [56:28] We are 56 minutes in.
    [56:33] And when these are published,
    [56:36] you will be able to find it in,
    [56:40] I need to close, way too many tabs are open.
    [56:48] You will be able to find it here.
    [56:50] This is where Cybertron.GitHub.IO.
    [56:56] There, shall I proceed with the branch creation?
    [57:01] Yes, as far as we have verified the results
    [57:05] before committing and the report makes sense.
    [57:10] Also make sure that you're going to use
    [57:13] the anti-slop scale.
    [57:16] To avoid the sloppy AI language in the report.
    [57:25] So I created a scale that tries to
    [57:33] basically remove these common vocab and terminology
    [57:39] that you often see in reports
    [57:43] or like long form of text of agents.
    [57:46] And then you can also use the
    [57:46] The only reason that is, is that, I mean,
    [57:48] obviously we all know we're using agents.
    [57:50] The only problem is that it oftentimes actually makes it
    [57:54] really vague.
    [57:55] Like not because of X, because of Y.
    [57:58] Oh, like, are you just using that just because
    [58:02] you don't have a better reference.
    [58:05] So anyway, but that's what some of the skills are.
    [58:08] You know, it's really useful.
    [58:10] So let's see.
    [58:11] The key issue is fine.
    [58:12] Look at that.
    [58:12] It did.
    [58:13] It did.
    [58:13] It did write garbage.
    [58:15] It did.
    [58:16] Sometimes it does write garbage.
    [58:18] So I'm glad we caught it.
    [58:19] It's getting to correct it.
    [58:22] And it usually does become a lot more readable
    [58:26] when you try to remove that language.
    [58:30] And then one of the other things I'm going to ask the agent
    [58:33] to do is create a summary set of messages
    [58:39] to post to the Telegram.
    [58:42] And.
    [58:57] I don't know why it changed a bunch of headers.
    [59:01] Oh, it changed it because it was in that week's report.
    [59:04] Okay, okay, fair, fair.
    [59:07] All right, I think that this is going to be one hour.
    [59:12] I'm going to try to keep it one hour.
    [59:15] I think we're going to go over a bit one hour,
    [59:18] but that's pretty much it for this week.
    [59:20] I will share the findings
    [59:22] and you will be able to see the report.
    [59:25] I hope this was helpful.
    [59:26] I hope this was useful by any means.
    [59:29] If not, feel free to ask away questions.
