#!/usr/bin/env python


# observation, state transfer happens initially (evolution is eventually non-random so will eventually diverge)
# according to the birthday paradox the probability that you share a "birthday" with another person in a popluation
# of n people in a year with d days is 1-((d-1)/d)**n 
#
# we can use this estimate the number of matches expected for prefixes of a given length
#
# assume that there are l layers, each with k options

l=15
k=2
n = 300

for m in range(1, l):
    d = k**m
    prob = 1-((d-1)/d)**n
    print(f"{m=}, {prob=}")

# observe that l doesn't matter, only k and n do
# it is better for our system to have a NAS with a larger number of small choices to increase the probablity of longer transfers
