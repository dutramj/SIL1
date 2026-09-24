# 3rd party libraries
import pandas as pd
import colorama

# VAHSimulator library
import vahsimulator as vah

colorama.init(autoreset=True)

ref_df = pd.read_csv(".\\records\\simulation_data_ref.csv")
exp_df = pd.read_csv(".\\records\\simulation_data_0.csv")

if ref_df.equals(exp_df):
    print(f"{colorama.Fore.GREEN}*** DataFrames are identical ***")

else:
    print(f"{colorama.Fore.RED}*** Differences were found ***")

    # Shape check
    if ref_df.shape != exp_df.shape:
        print(f"{colorama.Fore.YELLOW}Shape differs:")
        print(f"  ref: {ref_df.shape}")
        print(f"  exp: {exp_df.shape}")

    # Column check
    if not ref_df.columns.equals(exp_df.columns):
        print(f"{colorama.Fore.YELLOW}Column mismatch:")
        print("Only in ref:", set(ref_df.columns) - set(exp_df.columns))
        print("Only in exp:", set(exp_df.columns) - set(ref_df.columns))

    # Value comparison (only if shapes and columns match)
    if ref_df.shape == exp_df.shape and ref_df.columns.equals(exp_df.columns):

        diff_mask = ref_df.ne(exp_df) & ~(ref_df.isna() & exp_df.isna())

        if diff_mask.any().any():
            rows, cols = diff_mask.to_numpy().nonzero()

            print(f"{colorama.Fore.YELLOW}Total differing cells: {len(rows)}")

            for r, c in zip(rows[:10], cols[:10]):
                col = ref_df.columns[c]
                print(
                    f"Row {r}, column '{col}': "
                    f"{colorama.Fore.CYAN}{ref_df.iloc[r, c]} "
                    f"{colorama.Fore.WHITE}-> "
                    f"{colorama.Fore.MAGENTA}{exp_df.iloc[r, c]}"
                )

            if len(rows) > 10:
                print("... more differences omitted")

vah.plot_all(ref_df, exp_df)
