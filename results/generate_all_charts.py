"""
results/generate_all_charts.py

Runs both chart scripts back to back. Run from the project root:

    python -m results.generate_all_charts                 # RL charts only (fast, no dataset needed)
    python -m results.generate_all_charts --with-cnn       # RL charts + CNN charts (needs the
                                                             leaf-image dataset and, for the first
                                                             run, internet access for ImageNet weights)

Equivalent to running the two scripts separately:
    python -m results.generate_rl_charts
    python -m results.generate_cnn_charts
"""

import sys


def main():
    from results import generate_rl_charts
    print("=" * 70)
    print("RL / DQN irrigation model charts")
    print("=" * 70)
    generate_rl_charts.main()

    if "--with-cnn" in sys.argv:
        from results import generate_cnn_charts
        print("\n" + "=" * 70)
        print("CNN leaf-condition model charts")
        print("=" * 70)
        generate_cnn_charts.main()


if __name__ == "__main__":
    main()
