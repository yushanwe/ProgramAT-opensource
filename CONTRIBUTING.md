# Contribution Guidelines

Contributions to this repository will be primarily in one of three categories, as follows. 
Further description of each contribution class can be found in the sections describing their contribution requirements.

1. Visual Assistive Tools: these are ProgramAT tools meant to accomplish accessibility tasks.
2. Core codebase: these are contributions that modify the code for the server or mobile app itself.
3. Documentation: these are contributions that add or modify documentation, which could coorespond to either of the above contributions or be present on its own.

Each type of contribution has slightly different criteria for being accepted, detailed in their relevant sections below.

When making a contribution, potential contributors will be asked to mark the type of contribution their PR makes (this can involve selecting multiple categories if appropriate).
Contributions will be eligible to be accepted only once the approval criteria for all encompassed categories has been met.

Further, regardless of the contribution type, first-time contributors must sign the code of conduct. 
This can be done by checking the box in the PR template indicating you have read and agreed to the code of conduct. 
If you forget to do so, a configured Github action will prompt you to sign, and only then will a merge be successful.
This is only required for the first contribution, afterwards, your "signature" (Github username) will have been recorded as in agreement.

## Visual Assistive Tools
### Why do I want to make this contribution?
As mentioned earlier, visual assistive tool contributions intend to push visual assistive tools upstream.
You may want to make this kind of contribution for several reasons. Firstly, doing so makes a tool you've made and find useful available to the broader community so they can use it too.
Secondly, ProgramAT's structure is modular, so new tools may borrow functionality from existing tools on main. 
Thus, contributing visual assistive tools offers not just your specific tool to the community, but also allows future tools to leverage its component functionality.
By contributing your tool, you also support other users and developers in making tools of their own that share some aspect of functionality.

### What needs to happen for this contribution to be approved?
Since tools on main influence the development and behavior of other tools that come after it, it is very important that tools that make it upstream work properly.
To ensure this, we require 3 members of the community (other than the contributor themselves) to verify the tool works as it says it does.

This process starts by submitting a pull request containing your tool. Please give this pull request a name that reflects your tool's functionality. 
Once you have done so, the pull request is visible in non-production modes on the main server.

Tools can be reviewed by switching to "review mode" in the app, which connects you to the main server, where you can see all tools with associated pull requests.
Review mode is basically the same as dev mode, except you cannot create or iterate on tools directly there. 
Instead, you have the option to review a tool, which will bring you to an interface where you can select "yes, this tool works" or "no, this tool does not work", provide any other text based feedback you would like, and submit.
We request that these approvals be made from the app only, to encourage seriously testing a tool before making a decision on it.

Once a visual tool PR has three approvals (i.e. three submitted reviews of the "yes, this tool works" variety from distinct users other than the contributor), it is eligible to be merged upstream.
The exception to this is if a contribution is of multiple types, it does not become eligible to merge until _all_ relevant criteria or met, but three successful "this works" reviews satisfies the criteria for the visual assistive tool aspect of the contribution.

## Core code base
### Why do I want to make this contribution?
Contributions to the core codebase are designed to make changes to the underlying server or mobile app code.
If you would like to see a new feature in ProgramAT, or change how an existing feature works, this type of contribution may be of interest to you, as this is your opportunity to take implementation of things you would find helpful into your own hands.
If you encounter a bug in the app or server and believe you have fixed it on your end, this contribution type is the most appropriate for distributing this change broadly.

### What needs to happen for this contribution to be approved?
Core code base contributions are highly valuable: they are a big piece of how we expect ProgramAT to evolve and improve. 
However, since they modify ProgramAT's underlying structure, they are also somewhat risky, and as such are subject to a more stringent process of review.

For this type of contribution to be accepted, you should first create an issue describing your proposed contribution and assign it the community-proposed label (alongside other labels that are appropriate).
This issue should clearly describe what you propose (e.g. feature and proposed implementation, bug and proposed fix, etc.). 
Community members can then react to the issue, giving a thumbs up react to indicate they would like to see this change and a thumbs down react to indicate that they do not want to see this change.
Once the issue has recieved an aggregate of positive three thumbs up (i.e. at least three more thumbs up than thumbs down), AND a comment from a member of the research team indicating the change is given the go-ahead upon reaching three votes, it is eligible for a pull request.
A list of research team members who are qualified to leave such a comment can be found here: (TODO: make list of who is on the research team and link it)
Contributors' reactions to their own issues will not count towards the total of net three positive reactions.

Once the reaction and research team approval have been met, you can submit an upstream pull request containing your proposed change. 
Please include a link to your approved issue in this pull request. Pull requests of the core code base contribution type without a linked issue will not be eligible for merging upstream.
In this pull request, you should detail how your submitted code changes address the plans laid out in your approved issue.
For fastest approval, this description should include not just what you changed, but the line numbers, file names, and function names where appropriate for the associated changes.
To be approved for merging upstream, the pull request needs to achieve at least three approving reviews, including at least one from a member of the research team.
To be counted towards this total, the body of the review comment should contain the text "CORE CODE BASE REVIEW" (case sensitive). 
If a review is for a contribution of multiple types, it can contain multiple phrases, to satisfy both counts.
Contributors are not eligible to be a reviewer on their own pull request.

You may recieve a review that requests additional changes: if you recieve such a review from a member of the research team, you must make these changes regardless of outside number of acceptances before merging, and recieve an approving review from the same research team member before proceeding to merge.
You do not, however, need to regain approval of the issue, nor start from scratch in terms of number of prior approvals after submitting the required changes.

After meeting these criteria, the contribution is eligible to be merged upstream.
Again, the exception to this is if a contribution is of multiple types, it does not become eligible until criteria for _all_ types are met, but the box for the core code base contribution will be checked.

## Documentation
### Why do I want to make this contribution?
Documentation is a very important part of the health of any codebase.
You may want to make a contribution adding new documentation when submitting a contribution about the core code base or a new visual assistive tool, to make sure it is appropriately understood.
You may also want to make a contribution adding new documentation outside of submitting new code if you notice an area where documentation is absent or underspecified.
You may want to make a contribution updating existing documentation as well: for instance, if you notice existing documentation has become out of date, a documentation contribution empowers you to update it.
Similarly, if you notice conflicts within existing documentation, a documentation contribution is the appropriate way to resolve the conflict.

### What needs to happen for this contribution to be approved
To make a documentation contribution, you should submit a pull request containing the changed documentation, and assign the pull request the documentation label.
If the pull request is solely for documentation, this should be present in the PR title. 
If a PR contains multiple types of contributions, of which documentation is just one, however, this titling criteria does not apply 
(i.e. the title of a PR contributing both a visual assistive tool and documentation should follow the rule of titling the PR for what the tool does, not for being documentation).

To be approved for merging upstream, the pull request needs to achieve at least three approving reviews.
To be counted towards this total, the body of the review comment should contain the text "DOCUMENTATION REVIEW" (case sensitive).
If a review is for a contribution of multiple types, it can contain multiple phrases, to satisfy both counts.
Contributors are not eligible to be a reviewer on their own pull request.

For this contribution type, a review from a member of the research team is not strictly required, though it still may occur. 
In this vein, you may recieve a review that requests additional changes: if you recieve such a review from a member of the research team, you must make these changes regardless of outside number of acceptances before merging, and recieve an approving review from the same research team member before proceeding to merge.
You do not, however, start from scratch in terms of number of prior approvals after submitting the required changes.

Once you have achieved three approvals, without any unresolved requested changes from members of the research team, your documentation contribution is eligible to be merged upstream.
Again, the exception to this is if a contribution is of multiple types, it does not become eligible until criteria for _all_ types are met, but the box for the documentation contribution will be checked.



