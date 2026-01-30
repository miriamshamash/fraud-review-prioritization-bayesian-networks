import unittest
import argparse
from tests import TestProbQuery
from tests import TestEnumerateAll
from tests import TestEnumerateAsk

from tests.custom_test_runner import run_tests_with_custom_runner

if __name__ == "__main__":
    # Define the mapping of question names to test classes
    tests = {"q1": TestProbQuery, "q2": TestEnumerateAll, "q3": TestEnumerateAsk}
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run unit tests with optional question filtering')
    parser.add_argument('-q', '--questions', nargs='*', 
                       choices=tests.keys(), 
                       help='Specify which questions to test (e.g., -q q1 q2). If not specified, all tests will run.')
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.questions:
        # Run only the specified questions
        selected_tests = [tests[q] for q in args.questions if q in tests]
        if selected_tests:
            print(f"Running tests for: {', '.join(args.questions)}")
            result = run_tests_with_custom_runner(*selected_tests)
        else:
            print("No valid questions specified. Available options:", tests.keys())
    else:
        # Run all tests if no -q argument is provided
        print("Running all tests...")
        result = run_tests_with_custom_runner(*tests.values())

    # If you want a more terse output, you can run this
    #unittest.main(verbosity=2)
