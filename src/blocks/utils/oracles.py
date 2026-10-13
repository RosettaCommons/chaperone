import os

# ##################################################
# # Other
# ##################################################


# def renumber_residues_increasing_across_chains(
#     input_path: os.PathLike, output_path: os.PathLike
# ) -> None:
#     """This function takes in a PDB file and renumbers the res_ids such that they are monotonically increasing, even across chains.
#     This is necessary for using the AF-multimer PPI script at /mnt/net/software/containers/users/drhicks1/colabfold_ig/AlphaFold2_initial_guess_multimer.py
#     """
#     atom_array = load_any(input_path)
#     num_res = len(struc.get_residue_starts(atom_array, add_exclusive_stop=False))
#     res_id_renumbered = struc.spread_residue_wise(atom_array, np.arange(num_res) + 1)
#     struc.spread_residue_wise
#     atom_array.set_annotation("res_id", res_id_renumbered)
#     pdb_string = to_pdb_string(atom_array)
#     with open(output_path, "w") as f:
#         f.write(pdb_string)


# def _get_df_from_silent_scorefile(scorefile):
#     with open(scorefile, "r") as f:
#         lines = [line.replace("SCORE:", "").strip() for line in f if line.strip()]

#     # Convert cleaned lines into a single string for pandas
#     cleaned_data = "\n".join(lines)

#     # Read into DataFrame using whitespace as the separator
#     df = pd.read_csv(io.StringIO(cleaned_data), delim_whitespace=True)

#     return df


def is_ccd_name(name):
    """Checks if the name is a CCD code based on whether the path exists"""
    from pipelines import get_setting_field

    ccd_path = get_setting_field("CCD_MIRROR_PATH")
    path = os.path.join(ccd_path, name[0], name, f"{name}.cif")
    return os.path.exists(path)
