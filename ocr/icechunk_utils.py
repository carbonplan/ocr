from __future__ import annotations

import typing

import icechunk

from ocr.console import console

if typing.TYPE_CHECKING:
    import xarray as xr


def get_commit_messages_ancestry(repo: icechunk.Repository, *, branch: str = 'main') -> list:
    commit_messages = [commit.message for commit in list(repo.ancestry(branch=branch))]
    # separate commits by ',' and handle case of single length ancestry commit history
    split_commits = [
        msg
        for message in commit_messages
        for msg in (message.split(',') if ',' in message else [message])
    ]
    return split_commits


def region_id_exists_in_repo(repo: icechunk.Repository, *, region_id: str):
    region_ids_in_ancestry = get_commit_messages_ancestry(repo=repo)

    # check if region_id is already in icechunk commit history
    if region_id in region_ids_in_ancestry:
        return True
    else:
        return False


def insert_region_uncoop(
    session: icechunk.Session,
    *,
    subset_ds: xr.Dataset,
    region_id: str,
):
    import icechunk

    console.log(f'Inserting region: {region_id} into Icechunk store: ')

    while True:
        try:
            subset_ds.to_zarr(
                session.store,
                region='auto',
                consolidated=False,
            )
            # Trying out the rebase strategy described here: https://github.com/earth-mover/icechunk/discussions/802#discussioncomment-13064039
            # We should be in the same position, where we don't have real conflicts, just write timing conflicts.
            session.commit(f'{region_id}', rebase_with=icechunk.ConflictDetector())
            console.log(f'Wrote dataset: {subset_ds} to region: {region_id}')
            break

        except icechunk.ConflictError:
            console.log(f'conflict for region_commit_history {region_id}, retrying')
            pass
