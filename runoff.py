import random

def simulate_runoff_election(candidates, voters, num_voters):
    def count_votes(ballots):
        vote_count = {candidate: 0 for candidate in candidates}
        for ballot in ballots:
            vote_count[ballot[0]] += 1
        return vote_count

    def eliminate_candidate(ballots, candidate):
        new_ballots = []
        for ballot in ballots:
            new_ballot = [c for c in ballot if c != candidate]
            if new_ballot:
                new_ballots.append(new_ballot)
        return new_ballots

    ballots = voters.copy()
    
    while True:
        vote_count = count_votes(ballots)
        total_votes = sum(vote_count.values())
        
        print("\nCurrent standings:")
        for candidate, votes in vote_count.items():
            percentage = (votes / total_votes) * 100
            print(f"{candidate}: {votes} votes ({percentage:.2f}%)")
        
        # Check for a winner
        for candidate, votes in vote_count.items():
            if votes > total_votes / 2:
                return f"{candidate} wins with {votes} votes ({(votes/total_votes)*100:.2f}%)"
        
        # Eliminate the candidate with the fewest votes
        candidate_to_eliminate = min(vote_count, key=vote_count.get)
        print(f"\nEliminating {candidate_to_eliminate}")
        candidates.remove(candidate_to_eliminate)
        ballots = eliminate_candidate(ballots, candidate_to_eliminate)
        
        if len(candidates) == 1:
            return f"{candidates[0]} wins as the last remaining candidate"

# Example usage
candidates = ["Alice", "Bob", "Charlie", "David"]
num_voters = 100

# Generate random voter preferences
voters = []
for _ in range(num_voters):
    voters.append(random.sample(candidates, len(candidates)))

print("Initial candidates:", candidates)
print("Number of voters:", num_voters)
result = simulate_runoff_election(candidates, voters, num_voters)
print("\nFinal result:", result)