import os

# ##################################################
# # Other
# ##################################################


# def renumber_residues_increasing_across_chains(
#     input_path: os.PathLike, output_path: os.PathLike
# ) -> None:
#     """This function takes in a PDB file and renumbers the res_ids such that they are monotonically increasing, even across chains.
#     This is necessary for using the AF-multimer initial-guess PPI script.
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
    """Recognize CCD identifiers without requiring a site-specific mirror.

    A configured mirror may contain newer components than Biotite's bundled
    dictionary. Check it first, then fall back to the installed dictionary.
    """
    from biotite.structure.info import all_residues
    from chaperone import get_setting_field

    if not isinstance(name, str) or not name:
        return False
    ccd_path = get_setting_field("CCD_MIRROR_PATH", None)
    if ccd_path:
        path = os.path.join(ccd_path, name[0], name, f"{name}.cif")
        if os.path.isfile(path):
            return True
    return name in all_residues()
