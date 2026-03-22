# Sutro Group Meeting #9 (16 Mar 2026)

**Date:** 2026-03-16
**Presenter:** Yaroslav Bulatov
**Video:** [YouTube](https://www.youtube.com/watch?v=vdQ3NkEiOt8)
**Length:** ~60 minutes
**Agenda:** [Google Doc](https://docs.google.com/document/d/1B9867EN6Bg4ZVQK9vI_ZqykZ5HEtMAHJ7zBGGas4szQ/edit?tab=t.0)

## About

Yaroslav presents the Sutro Group roadmap, introduces the DMC metric, reviews Yad's automated research agent, and walks through GF(2) Gaussian elimination verification.

## Chapters

| Time | Topic |
|------|-------|
| 0:00 | Pre-meeting chat (Yad's agents in Telegram) |
| 7:01 | Agenda overview, motivation for the group |
| 7:57 | Why we're here: energy waste in AI training |
| 8:36 | Public domain vs open source, why it matters |
| 9:15 | Meeting structure and agenda document |
| 9:54 | Yad's automated researcher: what it does |
| 10:58 | How to use it: git clone, cd, run claude |
| 12:44 | Keep experiments small (Spark 7 compute budget) |
| 13:39 | Playing Yad's 5-minute demo video |
| 15:54 | Demo: EGD experiment on sparse parity |
| 16:35 | Side-by-side: Claude Code vs Gemini CLI |
| 19:54 | Results: EGD took 9 min end-to-end, Gemini struggled |
| 22:02 | Workflow reflection: 12 min active, need better notifications |
| 23:03 | What is the job of humans when agents do the work? |
| 25:24 | Verifying results: GF(2) was 1000x faster, is it real? |
| 27:56 | Digging into GF(2): why Gaussian elimination works |
| 28:06 | Sparse sum as solving linear equations |
| 29:56 | Gaussian elimination walkthrough on the whiteboard |
| 31:10 | Sparse parity as linear equations over GF(2) / mod 2 |
| 31:56 | Interactive visualization (AI Studio) for sanity checking |
| 33:01 | Hardware lock-ins: motivation slides |
| 33:42 | Examples: square airplane windows, QWERTY, VHS vs Betamax |
| 34:59 | Academic publishing and peer review as legacy system |
| 35:22 | Fossil fuels vs solar: legacy infrastructure discussion |

## Related

- [Meeting #9 Notes](../google-docs/meeting-9-notes.md)
- [Meeting #9 Agenda](../google-docs/meeting-9-agenda.md)
- [Yad's Session 1](sesh-1.md) (recorded same week)
- [EGD Experiment Findings](../findings/exp_egd.md)
- [GF(2) Findings](../findings/exp_gf2.md)
- [Yaroslav Verification Sprint](../google-docs/yaroslav-verification.md)

## Transcript

Transcribed with whisper-large-v3 via MLX (Apple Silicon). 962 segments, 35,690 characters.

??? note "Full transcript (click to expand)"

    [00:00] Progress.
    [00:32] You have a new hot channel in the Telegram.
    [00:35] Which one?
    [00:35] Chat, yeah, yeah.
    [00:37] Oh, that's right, Chat.
    [00:40] And he has his agents integrated with it,
    [00:43] I remember there were times, I was telling,
    [00:46] Hey, yeah, can you do this, and the agent would do it.
    [00:48] Are the agents in the Chat?
    [00:50] Well, they read, but they don't see,
    [00:52] so the agent would just do it,
    [00:54] and to tell Chat, Hey, just do the thing,
    [00:56] and Chat is like, What did you do,
    [00:57] it's like, Oh, you do the thing,
    [00:58] never to watch,
    [00:59] so to all the agents,
    [01:01] So it's kind of cool.
    [01:02] My human agents are removing themselves from the loop.
    [01:06] I'll just be talking to AI agents pretty soon.
    [02:19] Maybe I can be a .
    [04:25] The fail on that is that I have something to give thanks to.
    [04:27] So, let's see if I can make it harder.
    [04:29] Oh, I didn't mean to say that.
    [04:31] Nothing.
    [04:32] Yeah.
    [04:33] If you're all solo, pretty much have to go to the market.
    [04:39] I think it's at the, you know, I don't know what I'm going to do.
    [04:42] If you had a couple, if you had another employee or something,
    [04:45] then you might have to keep your product out of the way.
    [04:47] That's one thing.
    [04:48] You don't want to say, you know, just keep talking to them.
    [07:01] All right.
    [07:01] I posted in the Telegram channel, I posted the agenda.
    [07:05] So, I will, it's a high outline of the agenda.
    [07:11] It's meant to be asynchronous, but I'll cover some of those points in person.
    [07:17] But the point is, a lot of the stuff can be done asynchronously.
    [07:21] The point of this meeting is maybe to put names to faces and eat pizza.
    [07:27] So, I'll, the plan is, the pizza is coming in half an hour.
    [07:35] So, I'll plan to talk for maybe 15, 20 minutes.
    [07:40] So, okay, so the outline.
    [07:41] Why are we here?
    [07:42] So, we're here.
    [07:44] The reason we're here is in this document.
    [07:47] Sucha group.
    [07:48] Top level.
    [07:49] I should probably say.
    [07:57] And it has the motivation.
    [07:59] And the motivation is right now I'm seeing a lot of energy wasted.
    [08:04] And we want to fix it.
    [08:05] But also fix it in a way which maximizes benefit to humanity.
    [08:09] So, this last part actually is what explains some of the, explains the reasons why we do some of the things.
    [08:19] We do.
    [08:19] So, for instance.
    [08:21] We have this telegram group where everything is, it's public.
    [08:26] Anybody can export the entire archive.
    [08:29] Because I feel like if things are transparent, it's actually faster to move and easier to provide benefit.
    [08:36] And also every, all the code we're doing, I'm assuming is public domain.
    [08:40] So, public domain is actually much more important than open source.
    [08:44] So, public domain means somebody can take your code and make a lot of money on it.
    [08:51] And also it means somebody could use your code and not even credit you.
    [08:55] And I personally found like the crediting one, sometimes it's like annoying to think.
    [09:01] If you're using code and you're like, oh man, what are all the people I have to credit.
    [09:06] And it kind of adds friction.
    [09:07] So, the point is for maximum benefit to humanity, we want to have the least friction.
    [09:11] So, everything is open and public domain.
    [09:15] So, this document has meetings.
    [09:17] And we have this agenda.
    [09:19] So, okay, so motivation.
    [09:23] So, motivation is we're running deep learning.
    [09:27] The methods we invented were created during the times when we had CPUs.
    [09:33] Since then, we started using GPUs.
    [09:36] But a lot of the concepts didn't get updated, which adds friction.
    [09:40] So, this is one of the reasons things are bad, I believe.
    [09:47] So, where is my agenda?
    [09:54] Okay, done with motivation.
    [09:56] Now I'll talk about YAD stuff.
    [09:58] Last time, YAD was the last item and we didn't get to it because pizza arrived.
    [10:04] So, this time I'm moving it up.
    [10:06] So, YAD is a former SPC member and he had been really in the program synthesis.
    [10:12] And he made this automatic researcher report.
    [10:16] And I started using it last week and it's actually surprisingly good.
    [10:21] So, what it does, you can go to just the GitHub in the agenda and it gives some stuff.
    [10:33] So, he ran 30 experiments, found something good there.
    [10:38] And initially, I was like, okay, what do I do with it?
    [10:41] There's like to do, like do I edit the to do?
    [10:44] And YAD is like, no, no, no, you don't touch anything at all.
    [10:47] Don't touch the code, just talk to it.
    [10:49] Which feels a little bit weird.
    [10:51] Like, what if it doesn't do what you want?
    [10:53] Are you supposed to like ask it again nicely?
    [10:56] But it turned out it just worked out of the box.
    [10:59] So, the way you do it, you just do a git clone, ucd, and then you do run cloud with this flag, this syntax.
    [11:08] You can figure it out.
    [11:10] Yeah.
    [11:11] And then basically you get to this viewpoint.
    [11:13] Then first thing you can ask like what is this about?
    [11:18] Or like how do I, wait, I have like three microphones.
    [11:24] How do I run experiments?
    [11:28] So, you can basically interact with it in human language.
    [11:38] Anyway.
    [11:38] So, you should, I will not take your time, but you should just go there, run it, and you can basically interact with it.
    [11:45] And you can say, if you have an idea for an experiment, say, why don't you look up some other technique and try it.
    [11:53] Okay.
    [11:53] So, and now I can say, what is the top performing two methods in your experiments?
    [12:01] How good are they?
    [12:02] And do you have any ideas how to improve on them?
    [12:08] So, basically this is like having an LLM, but it also can run things, which is kind of cool.
    [12:15] So, it can fact check itself.
    [12:19] So, this is using Cloud Opus.
    [12:21] Maybe there's a faster LLM.
    [12:26] But the bottom line is you interact with it, you instruct it as if you instructed some instance.
    [12:31] And they do things for you.
    [12:33] All right.
    [12:33] So, GF2.
    [12:37] Yeah.
    [12:38] So, basically the experiments run half of a millisecond.
    [12:42] So, these are pretty fast experiments.
    [12:44] And this is a reasonable time frame to keep your experiments up.
    [12:47] Because most of the methods were invented by before the end of the 1980s.
    [12:52] And the hardware we had in 1980s was much slower than your laptop.
    [12:57] So, if they could do it on, you know, Spark 7, you can do it on one millisecond on your laptop.
    [13:02] It's very easy to get seduced.
    [13:04] into running bigger and bigger experiments but initially it's exciting
    [13:09] you're running more realistic problems but then it slows down your iterations
    [13:12] in the long term you will regret it so I think my job here is to remind everybody
    [13:16] humans and agents to keep experiments very small so yes so this is yet stuff
    [13:26] I'm still figuring out what it can do like I have a feeling at some point it
    [13:30] will stop working like I it just seems it would be too magical if it just went
    [13:35] all the way to another GPT and then the real work will begin but so far it's done
    [13:39] everything I've asked it to do and now we will play a five-minute video that
    [13:44] the ad recorded for us I wanted to walk you through an update for future yellow
    [13:51] and you just you don't clear what it really is and the simplest way possible
    [13:57] is that
    [14:09] search research so there is really there is nothing fat you just be very clear
    [14:15] what it really is in the simplest way possible is that the structured
    [14:19] workspace that allows coding agents to become a category of research agents or
    [14:23] rather a category of group of search research agents so there is really
    [14:26] there's nothing fancy going on other than being able to structure it in a
    [14:30] way that the coding agent can navigate it and then it would be able to perform
    [14:33] these tasks that we have so roughly that's what a diagram is over here that
    [14:36] you're getting to see okay so what we're getting to do today is that we're getting
    [14:39] getting getting getting getting getting getting getting getting getting getting
    [14:42] getting getting getting getting getting getting getting getting getting getting
    [14:43] getting getting getting getting getting getting getting getting getting getting
    [14:47] getting getting getting getting getting getting getting getting getting getting
    [14:47] getting getting getting getting getting getting getting getting getting
    [14:49] getting getting getting getting getting getting getting
    [14:51] getting getting getting getting getting getting getting
    [14:54] To sort of highlight what we are going to demo, we're going to try to have our coding agent perform a task and try to see how much work it takes.
    [15:04] What are some of the unexpected things? Was it easy? And what are the main takeaways?
    [15:09] So I initially had a task in mind. However, the agent have already performed that and already has done a write up.
    [15:16] So alternatively, we're going to try using this egalitarian gradient descent approach to try to drop SGD to potentially perform the sparse parity and the sparse sum below 10 milliseconds.
    [15:30] There is a lot more context to this. If you go back, the agent have done a whole write up on the to do.
    [15:34] And then there's a whole section of discoveries and context to this problem.
    [15:39] And whenever these results get published, you can actually observe them on this public website called ChangeLog.
    [15:44] So that's that's a lot of time.
    [15:46] I'm just going to right now jump in and then try to perform this sort of task.
    [15:51] So this dangerously skip permission and team mode, dangerously skip permission might be a little bit misleading because it does not really dangerously skip permission.
    [16:00] It's a lot more like allowing the agent to do a bit more experimentation in this scenario.
    [16:04] OK, so there are two things we can do. One of them is that I oftentimes actually just pop in here because this is connected to the telegram and the Google Doc.
    [16:12] I can ask the question, hey, what are the most recent updates?
    [16:16] That the group is talking about specifically in the prosperity challenge channel and the chat yet.
    [16:23] So so this is sort of my way of trying to just see what's going on.
    [16:28] Spark. OK, I guess there is going to be text to speech errors.
    [16:31] That's one of the unexpected errors that we are going to have.
    [16:35] OK, so while that's happening, I'm going to also launch the Gemini CLI over this site.
    [16:41] So the dash dash yellow is the equivalent of dangerously skip permission.
    [16:47] Just to kind of give you a bit a bit of explanation of what's going on.
    [16:50] So, OK, so this guy is kind of getting confused over here.
    [16:52] So I might go ahead and ask, OK, I guess it's figuring it out itself.
    [16:56] So the skip permission part is really not not a big issue in this example that we are in.
    [17:03] And the main reason that is, is because we're not really mutating production data.
    [17:07] And because we are versioning the data, even if it breaks things, we can bring it back.
    [17:10] OK, so I'm going to ask the agent to perform this task.
    [17:15] So we're asking the Gemini CLI to perform this task, which is a different repo.
    [17:19] Just to keep in mind, it's a clone of that repo, but this is the original version.
    [17:23] OK, so this gave us a heads up. It's talking about me saying that I'm going to do this video that I'm doing right now.
    [17:30] And OK, so I'm actually going to now just perform the same task over here.
    [17:35] So with the task is implement and test the illiterate gradient descent on the sparse parity.
    [17:41] See if it breaks 10 milliseconds. OK, so what I want you to pay attention to.
    [17:45] Is that two things. A my coding agent specifically for cloud code.
    [17:50] As you can see, it's actually really annoying to do this with Gemini.
    [17:53] I make it verbose. So it actually tells me everything it is doing.
    [17:57] So this is actually really important to understand how this coding agent would function as a research agent.
    [18:02] So that's like I'll say the first step. The second step is that.
    [18:07] This these sort of skill sets are actually things that you end up creating as you go along in order to reduce the steps.
    [18:15] That the agent takes in order to perform that. OK, so the agent came back to me.
    [18:18] It says I have that. It says what framing do you want?
    [18:21] OK, so it's giving us options. The project already has EGD listed as to do.
    [18:26] The strict reliability, gradient elimination, un-leaning towards two, first proof.
    [18:34] Sounds good. Let's try that out. In fact, let's try both of them and then let's compare results.
    [18:39] And then after that, let's try the sparse sum example to be compared.
    [18:43] But ideally, we're going to utilize the GPU because you want to measure the efficiency as you may know.
    [18:49] And keep in mind that you do have access to MotoLabs if you want to deploy it.
    [18:53] The most important part is to keep track of the results because we need to do a write up.
    [19:00] So that is basic. These things that I'm saying are not just out of blue.
    [19:06] So, OK, actually, let's wait. Hold on. So look, what is this saying?
    [19:09] The user wants. So this is like really important. This is what you want.
    [19:12] The coding agent to do this is what really makes it a research agent that it takes this complicated, fluffy language that I'm using and it tries to break it down into a set of task lists and it tries to turn into like some order of operation.
    [19:26] And so so this is going to run the results. I'm going to probably pause it for the sake of time and then try to see what the results will be while it's going to do the implementation.
    [19:36] Also, all while that's happening. Unfortunately, this is how long it takes for Gemini to run.
    [19:42] So I still like to perform that. This is not a good demo.
    [19:45] I should have tried Codex. I don't know what this is even doing. Configuring.
    [19:48] OK, this is bad. All right. So this is good over here.
    [19:54] This is the thinking trace. This approach is recommended.
    [19:57] Look at that. It already has even told me. Let's go with a. All right.
    [20:00] I'm going to pause this time and then I will come back when the results are out.
    [20:06] Hey, folks, I'm back. So the coding agent ended up finishing the experiment of applying the egalitarian.
    [20:12] GD variant or it's actually EGD to the sparse parity and the sum.
    [20:19] Both of them. The agent itself took about nine minutes not only to run the experiments, but also to do verification reasoning.
    [20:28] Do the deployment on motor labs to do the GPU, do a whole write up.
    [20:33] And also so altogether experiment and writing solution and adding a new problem set took about nine minutes.
    [20:40] That is for cloud code.
    [20:42] Unfortunately, it did not end up working out too well for Gemini CLI.
    [20:47] It took not only longer, but it went into some dark hole, but it ended up switching into anti gravity.
    [20:54] And so that we also have some results that we can look at anti gravity.
    [20:58] I will explain this a bit more later. I think right now I want to focus on answering the questions that the main three things which were how long it took.
    [21:06] Where were like some of the unexpected things and what we learned and what is the outcome?
    [21:11] So that's that.
    [21:12] So I'll put this in the background.
    [21:12] But for now, the main things that we learned is that first of all, the effort only took about putting in really a couple of prompts.
    [21:23] I guess you could call them prompts, but it was just me directing the coding agent to perform the task.
    [21:28] And then eventually, eventually I ended up here after nine minutes, which basically said I said, do you have a write up?
    [21:36] And what are the findings?
    [21:38] And it goes like, yes, I have a write up.
    [21:39] These are this is the document.
    [21:42] And then eventually and then I said, commit, deploy and help me write up a message for the telegram for the team to share with you all.
    [21:50] So overall.
    [21:53] If you were to measure the time, it did take an hour, but it was not really an hour.
    [21:58] It took about 12 minutes and a couple of back and forth to do this.
    [22:02] I ended up getting distracted, leaving my computer and coming back.
    [22:05] So it kind of teaches you that there needs to be a better infrastructure.
    [22:11] To have actually the coding agent communicate with you back and forth, have a proper notification system.
    [22:17] So when it completes for the agent to follow up with you.
    [22:20] But yeah, so I think like that's that's like one answer.
    [22:22] How long it took and what are like some of the the step process looks like in terms of the finding.
    [22:28] I guess there are a couple of interesting things here to look at.
    [22:32] But I think like the big finding here is that it did it did end up doing slower but higher accuracy.
    [22:42] The sub 10 millisecond was not achievable with the EGD and the EGD did solve the sparse sum where the standard SGD diverged.
    [22:51] So you can continue on your own to do.
    [22:54] So this is just to give a flavor of what work looks like, because before our work was to come up with algorithms.
    [23:03] The work is at a higher level and I'm still figuring out.
    [23:07] So, yeah, there's like way more advanced than me on this stuff.
    [23:11] I'm still figuring out how to insert myself in the process because like what to do when it doesn't work.
    [23:18] But for now, my work around if I don't know what to do, I just ask what should I do.
    [23:23] So if you if you ever want to try it out and you don't know what to do, I think you can just do git clone CD.
    [23:31] Then do close dangerous. Let's keep permissions.
    [23:33] And you can just say, how does this work?
    [23:37] How do I run experiments?
    [23:40] And basically just follow the instructions.
    [23:42] So you can ask it to teach you how to use it.
    [23:47] All right. So this is yet stuff.
    [23:49] So, yeah, I'm pretty. Yeah.
    [23:53] I think I know the answer.
    [23:54] But just how does a tool like this fit into the set of the law or perhaps the best part of the goal of the project?
    [24:04] Having an auto research agent because you really love something like me, which is not that type of goal.
    [24:09] To contribute to the.
    [24:13] I just want to.
    [24:14] Yeah. I mean, the goal is to get an agent which can go all the way to the GPT.
    [24:19] It's an agent which can train on the entire hundred million characters or to have an algorithm which can train on the entire hundred million characters of Wikipedia.
    [24:28] Hopefully, ideally by using an agent.
    [24:31] But I'm sure it will not.
    [24:34] If I tell it to do it, I'll get stuck.
    [24:36] I haven't tried it, but I don't do it.
    [24:39] It's a lot.
    [24:39] So that's why we're taking baby steps.
    [24:41] We're like trying to resolve prosperity and taking baby steps and doing careful checks.
    [24:47] So I can actually.
    [24:51] So this one is actually.
    [24:55] We used to be in a different room, but we have this one for the next two and a half months.
    [25:00] So every Monday here.
    [25:06] So one thing.
    [25:07] So the question becomes.
    [25:09] If the agents are doing the work, what is the job of humans?
    [25:13] There is a higher level applying our experience to tell it where to go.
    [25:16] But the second one agent can hallucinate.
    [25:19] They can like gaslight themselves.
    [25:21] So another role of humans is to actually check.
    [25:24] So this is when I looked at the other result.
    [25:27] When I looked at his results, his top result was like a thousand times faster.
    [25:34] So I'm like, OK, I have to check this one.
    [25:36] And it gives the Gaussian.
    [25:41] I don't know what the results are.
    [25:43] I did switchboard.
    [25:45] You go.
    [25:54] Oh, it's the old reboot.
    [25:57] OK, so there's some results.
    [26:02] This Gaussian elimination was a thousand times faster.
    [26:06] So this is in my agenda.
    [26:09] I think I think.
    [26:10] I think my job was to go in and verify it.
    [26:13] And then actually, when I was doing this, I was doing like this very meticulous everything.
    [26:19] What I'm doing.
    [26:20] So at 1 15, I was like trying to remember what I was doing.
    [26:24] And finally, I remember that by 1 45, I was actually started to do things.
    [26:29] So I'm recording this because I'm hoping later at some point in the future, an agent can read my box and they can reproduce it.
    [26:36] So instead of doing it myself, I'll just say, do what you already did last week.
    [26:41] But now.
    [26:43] So this is what I did.
    [26:44] This link from the agenda.
    [26:46] But I basically I opened his repo in anti-gravity and I opened agent manager.
    [26:54] It could also work in code.
    [26:55] And then basically I first said, well, why don't you run it?
    [26:59] Does it work?
    [27:01] And then I say, is it self-contained?
    [27:03] It was not self-contained.
    [27:05] I basically.
    [27:06] I just say, just turn it into a single Python file.
    [27:08] So I just wanted a single Python file that I can copy paste and I can ask my LLMs, which I trust more to see if it works or not.
    [27:18] So then I basically created this self-contained file.
    [27:21] And then I took the self-contained file and stuck it into deep thing.
    [27:25] And perhaps is it cheating?
    [27:26] Is it really doing it?
    [27:28] And said it was OK.
    [27:30] But also I think LLMs tend to tend to like flatter each other.
    [27:36] So if you generate the paper using LLM and you ask another LLM, is this paper legit?
    [27:41] They'll say this paper is totally legit.
    [27:43] So they can have like this bias.
    [27:45] So to go further, I basically started some more what it's actually doing.
    [27:51] And what it's doing is it's actually a pretty cool.
    [27:56] I think so actually a dug into.
    [27:58] So why?
    [27:59] Why is it doing this Gaussian elimination?
    [28:04] Yeah, Gaussian elimination.
    [28:06] So and the reason is that the sparse parity task is actually very similar to a sparse sum task.
    [28:12] So and as far as some some tasks, suppose you have, for instance, five bits, you know, X1, X2, X3, or four bits.
    [28:21] And maybe only two of them are valid.
    [28:25] Maybe it's like this one and this one.
    [28:29] So this is a so you give it a string and then you look at the sum of the hidden bits.
    [28:35] So for instance, if you give a string for the one zero zero one, the sum will be the first one and the third one.
    [28:43] It will be two.
    [28:45] And you can look at this problem as the problem of estimating of solving a linear equation.
    [28:54] So you can you basically have something like W1X1 plus W2X2 plus W3X3.
    [29:08] And then you have some hidden formula.
    [29:12] And then you have some sample strings, maybe like this is zero one zero zero one.
    [29:19] And then you have labels and then you're given various things like this.
    [29:25] This will be like also one.
    [29:29] But this one.
    [29:31] This one.
    [29:34] This one will be zero.
    [29:36] So in a sparse sum task, you would be given strings and then you know their sums.
    [29:43] And the sparse sum task is actually known as the task of solving linear equations.
    [29:49] So you have some linear equations and you have to solve for the coefficients.
    [29:56] And if you look how you're solving them, there's this technique of Gaussian elimination.
    [30:01] And I'll just give a flavor of how it works.
    [30:06] So a flavor is.
    [30:11] So suppose you have some equation like this.
    [30:17] You can, if it's a valid equation, then you can add or subtract these two equations and you still get a valid equation.
    [30:26] So you get an equation like this.
    [30:30] But you can also, if you do a subtract, no, if you subtract it, you get rid of x1.
    [30:37] So if you subtract one equation from another, the new equation you get will be 2x2 equals to minus 2.
    [30:52] So basically this is the core idea of how this Gaussian elimination works.
    [30:58] You create new equations and now if you have this equation, now you can solve for the unknown and then you can back substitute.
    [31:07] So this is known as Gaussian elimination.
    [31:10] And sparse parity, there's a thing called, you introduce a different arithmetic where you're saying it's like 0 plus 1 equals 1.
    [31:19] And then 1 plus 1 equals 0.
    [31:22] It's known as like mod 2 arithmetic.
    [31:25] And it turns out in this new kind of arithmetic.
    [31:28] Similar ideas work.
    [31:30] So this sparse parity task is basically, it can be viewed as solving linear equations over a different field.
    [31:38] So and as a part of my verifications script, I'm also using AI to help me double check if things work.
    [31:49] So one tool I like is, I think Michael used, you deployed an app on plot code.
    [31:56] What was the thing you did with the peddling?
    [31:59] Yes, plot code.
    [32:00] I do a plot code maybe app.
    [32:01] And then you...
    [32:02] Yeah, it's just what it did as an artifact and plot code.
    [32:04] And then it's also a different thing.
    [32:05] Okay.
    [32:05] So it's an artifact and I've been using AI studio.
    [32:09] So I guess it's similar.
    [32:11] So in AI studio, I basically said something like, can you give me an interactive visualization to help me understand how Gaussian elimination works over mod 2.
    [32:24] So I did something like this.
    [32:26] And it thought for two.
    [32:26] Two minutes.
    [32:27] And it gave me this app.
    [32:31] And basically I can walk step by step how this elimination works.
    [32:37] So this was basically part of sanity checking.
    [32:39] I took the solution.
    [32:41] I didn't trust that AI was not cheating.
    [32:43] And I basically dug in and made sure that this solution made sense, which gave me a bit more sanity.
    [32:50] Okay.
    [32:51] So that's all I want to say about the other stuff.
    [32:56] Any questions?
    [33:01] So the next five minutes I'll talk about motivation for this effort.
    [33:07] So why are we even...
    [33:08] Why are we trying to replay things from scratch?
    [33:14] And Marisa?
    [33:15] Check, check, check.
    [33:17] So I made some slides.
    [33:19] So the agenda has the notebooks.
    [33:21] And I'll show some of the slides that are relevant here.
    [33:28] Let's see.
    [33:28] I want to get one of these slides.
    [33:31] So basically in systems we have often examples when we have a legacy system and we kind of carry it over to new design.
    [33:42] So for instance, windows are square.
    [33:45] So when people design airplanes, they did the first airplanes with square windows.
    [33:50] And there were some air disasters.
    [33:52] And then they changed them around.
    [33:54] They had to do it effectively.
    [33:55] So I think that's the first thing.
    [33:55] Another example is QWERTY keyboard.
    [33:59] The reason was that the arms would get stuck.
    [34:02] So they redesigned.
    [34:03] So...
    [34:05] But it's actually less efficient.
    [34:06] So the work keyboards are more efficient.
    [34:08] But there's just too much momentum behind QWERTY.
    [34:11] So we're stuck using QWERTY even though it's electronic.
    [34:16] Another one is fireplace.
    [34:19] It used to be in the center of the house.
    [34:22] But now we're still designing our houses around the central ornamental fireplace even though we don't really need it.
    [34:32] I actually don't know what this is.
    [34:35] I'll come back and learn more about it in the future.
    [34:39] Also, I don't know what this one is.
    [34:41] I like this one.
    [34:42] So when VHS came out, Betamax was actually a smaller cassette.
    [34:50] But there was basically a load of it.
    [34:52] There was a lot of legacy VHS stuff.
    [34:53] So it was too hard to break in.
    [34:57] Academic publishing is an idea.
    [34:59] So why do we even have peer review in the first place?
    [35:02] We used to have limited paper.
    [35:04] So we had to decide who gets put on this new paper.
    [35:08] And we still carried over this concept.
    [35:12] And right now peer review is not that useful.
    [35:16] It's a bit of a crisis.
    [35:19] Carbon.
    [35:19] Actually, maybe you have an opinion on this.
    [35:22] This is a generated slide.
    [35:23] But it's telling me that society is deeply infringed on fossil fuels even though you could switch.
    [35:30] And actually, a friend of mine works for the Department of Energy.
    [35:32] And he says that solar power is actually already cheaper to use.
    [35:37] But there are some subsidies or some legacy reasons.
    [35:40] Do you have an opinion on this?
    [35:42] I mean, solar.
    [35:45] Usually when people say solar is cheaper than fossil,
    [35:50] they're usually saying that today the cost of the solar cells has come down so low
    [35:56] that you can generate the equivalent amount of electricity more cheaply than you can from burning fossil fuels.
    [36:06] So it's more about the primary energy or electricity generation.
    [36:10] Then you get into questions about CapEx and how solar is kind of free of an ongoing business in the century.
    [36:18] But then you have the paper.
    [36:20] But usually when someone says that they mean that the CapEx basis is not cheaper to build a solar power
    [36:27] than the coal and the coal and the carbon.
    [36:29] But that was not the case for a really, really, really long time.
    [36:32] It's only been a few years.
    [36:35] I see.
    [36:37] And actually, a cul-de-sac.
    [36:38] I learned it this morning.
    [36:40] Cul-de-sacs have this round shape to accommodate carriages.
    [36:45] So today a car can do a three-point turn.
    [36:48] It doesn't need a cul-de-sac, but we still left this design.
    [36:52] But this one is actually my favorite example.
    [36:54] It's a recurrent laryngeal nerve.
    [36:57] So we have a nerve which went from the brain to the gills.
    [37:02] But the way it's programmed during in-growing development,
    [37:06] it starts as in a fish and then the heart kind of drifts down.
    [37:10] And in mammals, it gets stuck and it kind of grabs this nerve and it goes around.
    [37:17] I really am really lucky.
    [37:18] I'm enjoying this high artistic quality I'm getting out of this.
    [37:24] Actually, another thing I learned, this is the longest cell in the history of life.
    [37:28] When we had a sauropod, the recurrent laryngeal nerve was 50 meters.
    [37:33] It's one cell.
    [37:37] And the problem is that it's kind of hard to fix.
    [37:39] So if you are, the nerve goes through the heart, so it's kind of, there isn't,
    [37:46] you can't incrementally fix it.
    [37:48] But if you try moving it, you get a non-working giraffe.
    [37:52] And what I'm saying here is motivation.
    [37:54] The current design here, we have a giraffe in our tributes.
    [38:00] And why should you believe it?
    [38:04] And the reason, let's see, put it right there.
    [38:08] Yeah, so, and I'm saying this and you should believe it
    [38:11] because I have been working on this for a long time.
    [38:15] So in meeting number one, I gave some background.
    [38:18] 2004 was my first paper.
    [38:20] But then in 2013, I led the first deep learning deployment at Google.
    [38:25] I was on the TensorFlow team.
    [38:27] I implemented the first, one of the first, basically, I did the first Python client.
    [38:35] I was working on back propagation.
    [38:37] Then I worked on gradient checkpointing at OpenAI where we reduced memory usage.
    [38:43] I also worked at, I trained transformers.
    [38:46] We beat Google at this image competition.
    [38:52] I implemented back propagation at least five different times in TensorFlow team, in Pythor's team.
    [38:59] So yeah, so basically, the way when I look at some of the choices we're doing,
    [39:04] I know where they came from.
    [39:06] And back in the time, they were quite arbitrary.
    [39:08] So when people right now think, well, why do we need backprop?
    [39:12] Is it fundamental?
    [39:13] At the time, that was just the thing that worked.
    [39:15] And I was sure we would do something else later.
    [39:18] But now people are so used to it, they forget there is any other alternative.
    [39:26] All right, so, okay, two more sections.
    [39:32] So I spent some time thinking about the future.
    [39:34] Like, what do we need to do to avoid failure?
    [39:37] So this is a famous example.
    [39:41] In 2016,
    [39:45] there was a data set of radiology images,
    [39:50] and Google trained a model which was giving accuracy better than humans.
    [39:55] And at the time, they said, okay, in two years, we will not have,
    [39:58] or in five years, we will not have radiologists.
    [40:01] And then Elon Musk was even more optimistic.
    [40:04] He thought maybe in 2017, we won't have software and cards.
    [40:07] But it didn't happen, and there's various analysis why it didn't happen.
    [40:12] And one of the issues is that people optimize for the wrong reasons.
    [40:15] So people picked this classification accuracy on the benchmark,
    [40:20] and they optimized it, but it didn't actually optimize for the actual objective,
    [40:25] which is being useful at the office.
    [40:28] So when I think of previous failures, this is actually the most important thing to get.
    [40:33] So this is something I spend a lot of time thinking about.
    [40:38] How do we make sure we get the right objective?
    [40:40] And then there's some other issues.
    [40:43] Risk management.
    [40:44] There's a segment where you get too into some theory without having a way of checking it quickly.
    [40:50] So to manage your risk, you always need to be estimating how far you want to go before giving up.
    [40:57] Self-optimization.
    [40:58] You often have, you divide the work across different teams,
    [41:02] and they each solve their individual problem, but together, the problem isn't solved.
    [41:08] Also, maybe you forecast future incorrectly.
    [41:12] Here, I'm focusing on GPUs.
    [41:14] Because I feel like GPUs will be around in 15 years.
    [41:18] One of the reasons is we're building for Stargate now.
    [41:21] Those data centers have a 15-year depreciation cycle.
    [41:25] So probably people will want to get their money's worth.
    [41:28] So we'll probably get some GPUs for at least 15 years.
    [41:33] Yeah, and the way I think about the progress on this project is we want to,
    [41:38] we're starting with three problems, and we want to get to GPT-2.
    [41:42] So that's one axis.
    [41:43] Getting better and better solutions.
    [41:47] And the other axis is the objective.
    [41:50] So right now, we looked at average reuse distance, but this is not the real objective.
    [41:57] I feel like the ultimate objective, which is not before, is the warm-to-the-touch objective.
    [42:02] If you have a good algorithm, you should be able to run it in your laptop and just touch the laptop.
    [42:06] If the laptop gets super hot, well, maybe your thing is doing too much.
    [42:11] So that can be, I think, viewed as a ground truth.
    [42:14] And then we can come up with a proxy objective.
    [42:17] Maybe looking at sensor readings, maybe looking at power usage, maybe looking at memory usage.
    [42:22] But ultimately, I think to avoid the radiology problem, you should be able to actually run it in your laptop and actually touch it.
    [42:31] Because that's an equivalent of being in the doctor's office and using the system in the doctor's office.
    [42:41] Speaking of objectives, we started with objective average reuse distance.
    [42:47] I did some research and I found actually a much better objective called data movement complexity.
    [42:55] Let's see how much I want to touch.
    [42:57] Yeah, I do want to touch the other one.
    [42:58] Oh, hello.
    [43:01] Actually, maybe I don't want to touch the other one.
    [43:03] So data movement complexity is in the agenda.
    [43:08] So it basically...
    [43:11] It's a much more sensitive measure of how you're using the memory and that you're using cache.
    [43:20] Yeah, that's ours.
    [43:23] What?
    [43:27] It's our pizza.
    [43:33] Oh, one, four, two, six?
    [43:36] One, four, two, six.
    [43:38] Sorry.
    [43:40] I didn't get the...
    [43:43] I didn't know that it was...
    [43:50] How is data movement complexity ?
    [43:55] It's just that it sounded like it was being incurred.
    [44:05] Yeah, okay.
    [44:06] We can...
    [44:07] Okay, I want to spend two minutes talking about the data movement complexity.
    [44:10] So basically, the data movement complexity, it's the idea is that you have...
    [44:19] When you're using list LRU cache, you can basically...
    [44:22] There's a heuristic you can keep most recently used bits in a cheap nearby cache, which is small.
    [44:31] And the bits which were not used for a while in an expensive one.
    [44:35] And it basically...
    [44:38] Imagine this scenario where you have this concentric rings and there's a very simple formula.
    [44:44] The cost of a memory X is the square root of the used distance, meaning how many bits ago it's been used.
    [44:53] And I think this is a good metric.
    [44:56] It's a good metric because there's research on it, there's papers on it.
    [44:59] And if you wanted to get plot to implement it, what you would need to do is just point it to this paper and say,
    [45:07] I want to use this.
    [45:08] And you can see that there's this DMC metric from this paper.
    [45:10] And that should be enough to implement it.
    [45:15] So I think for next week...
    [45:19] Oh, so one thing about planning.
    [45:21] One lesson I learned is that it's tempting to start making progress on several dimensions simultaneously.
    [45:27] But you usually regret it.
    [45:29] So I prefer to work on one axis at a time.
    [45:32] So this time there's a new metric.
    [45:35] So for the next homework we can continue the same problem.
    [45:38] But now we improve the metric.
    [45:41] And the metric is this new metric ARD.
    [45:44] And basically iterate on this process, getting agents to invent new methods which are faster.
    [45:56] Iterate on the new metric.
    [45:57] So the idea is we're keeping the problem fixed.
    [45:59] So the idea is every time we only change one at a time.
    [46:02] Either improve the metric or the problem.
    [46:04] So this time we're keeping the problem again for the third week.
    [46:08] But we're improving the metric.
    [46:09] And the metric is in the agenda, DMC.
    [46:13] And I have strong preference for this metric.
    [46:22] Data movement complexity.
    [46:25] I guess we can close it. Nobody will come.
    [46:28] So data movement complexity, the reason for this metric is that
    [46:32] it represents, if you actually, the memory that's nearby is cheaper, but it's smaller.
    [46:39] And the memory that's far away is more expensive, but it's larger.
    [46:44] And actually the cost is roughly proportional to the square root of the size.
    [46:51] And they actually did some, in the paper they have a graph that's very similar to this.
    [46:55] This is latency, but there is the same graph for energy,
    [46:59] where the distance between the two is the same.
    [47:02] And from the processor, there's this square root relation.
    [47:09] So basically, yeah, we can use this metric to estimate how much energy
    [47:13] it would be to access a specific data information.
    [47:19] Yeah, and I think this is also, I think, an interesting metric.
    [47:23] What makes the metric interesting is when it's a difficult metric.
    [47:28] It's a difficult metric to gain.
    [47:30] And I think this one is difficult to gain because it actually represents,
    [47:35] you could actually implement cache where this would exactly be this,
    [47:39] because if you, you could actually have actual wires going in this concentric circle like this,
    [47:46] and the length of the wire is proportional to like the square root of the area.
    [47:54] So the energy you needed to, you know,
    [47:59] to use a wire of length L is proportional to that length.
    [48:02] So you could actually conceive of a field, an actual processor,
    [48:05] which achieved that energy, and it would lose,
    [48:08] use like this much energy lost in heat.
    [48:14] So it's not, I don't think it's easy to gain.
    [48:17] If it's something that's man-directed to physics,
    [48:19] it probably can't be gained in a boring way.
    [48:22] It has to be something interesting.
    [48:27] Great, this is, let's see.
    [48:29] This is all actually fit in, fit in, yeah, actually fit in the agenda.
    [48:36] So with that, we can start the game.
    [49:02] Okay.
    [52:05] Thank you.
    [52:39] Thank you.
    [53:00] Thank you.
    [53:41] Thank you.
    [54:05] Thank you.
    [54:33] Thank you.
    [55:11] Thank you.
    [55:33] Thank you.
    [55:57] Thank you.
    [56:44] Thank you.
    [57:05] Thank you.
    [57:37] Thank you.
    [57:57] Thank you.
    [58:57] Thank you.
    [59:20] Thank you.
    [59:33] Thank you.
    [60:05] Thank you.
