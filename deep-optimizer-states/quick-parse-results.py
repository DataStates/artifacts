import re
import argparse

def parse_log(file_path):
    # Regular expression to match "TIMER:interval-time" entries and extract the time value
    pattern = re.compile(r"<TIMER:interval-time,([\d\.]+)>")

    # List to hold the extracted data
    times = []

    # Read the file and extract data
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))

    return times

def print_results(times, log_name):
    print(f"Interval times from {log_name}:")
    for i, time in enumerate(times, 1):
        print(f"Iteration {i}: {time} seconds")

def main():
    parser = argparse.ArgumentParser(description="Parse log files for TIMER:interval-time entries.")
    parser.add_argument("--vanilla-deepspeed", type=str, help="Path to the vanilla-deepspeed log file", required=True)
    parser.add_argument("--deep-optimizer-states", type=str, help="Path to the deep-optimizer-states log file", required=True)

    args = parser.parse_args()

    # Parse each log file provided as command line argument
    times_vanilla = parse_log(args.vanilla_deepspeed)
    times_optimizer = parse_log(args.deep_optimizer_states)

    # Print the results for each log file
    print_results(times_vanilla, "Vanilla Deepspeed")
    print("\n")  # Add a newline for better separation
    print_results(times_optimizer, "Deep Optimizer States")

if __name__ == "__main__":
    main()

