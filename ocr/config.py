from dataclasses import dataclass


@dataclass
class TemplateConfig:
    bucket: str = 'carbonplan-ocr'
    prefix: str = 'intermediate/fire-risk/tensor/TEST/TEMPLATE.icechunk'
    uri: str = None

    def init_icechunk_repo(self, readonly: bool = False) -> dict:
        """Creates / Opens an icechunk repo and returns the repo and session.

        Args:
            bucket (str, optional): aws bucket name. Defaults to 'carbonplan-ocr'.
            prefix (str, optional): bucket prefix. Defaults to f'intermediate/fire-risk/tensor/TEST/TEMPLATE.icechunk'.

        Returns:
            dict: dict containing a icechunk repo and icechunk session {'repo':icechunk.Repository, 'session', icechunk.Session}
        """
        import icechunk

        storage = icechunk.s3_storage(bucket=self.bucket, prefix=self.prefix, from_env=True)

        repo = icechunk.Repository.open_or_create(storage)

        if readonly:
            session = repo.readonly_session('main')
        else:
            session = repo.writable_session('main')
        return {'repo': repo, 'session': session}

    def wipe_icechunk_repo():
        # TODO: It might be nice to be able to wipe the template
        raise NotImplementedError("This functionallity isn't added yet.")

    def create_icechunk_zarr_chunked_template():
        # TODO: This should create a template, write it and commit it.
        # icechunk_repo_config = self.init_icechunk_repo()

        # TODO: / WARNING The chunking config of the end result may differ from the USFS one.
        # We need to make the ChunkingConfig more general most likely

        # TODO: Add template creation logic. Should the template chunk params be passed in here instead of inferred?
        raise NotImplementedError('TODO: Not complete')

    def check_icechunk_ancestry():
        raise NotImplementedError('TODO: Not complete')

    def __post_init__(self):
        # TODO: Make this more robust with cloudpathlib or UPath?
        self.uri = 's3://' + self.bucket + '/' + self.prefix


@dataclass
class BatchJobs:
    """Dataclass that generates and stores batch commands"""

    region_id: tuple[str, ...]
    run_on_coiled: bool = False

    def generate_batch_commands(self):
        batch_command_list = []
        for rid in self.region_id:
            if self.run_on_coiled:
                cmd_str = (
                    f'uv run coiled batch run --name {rid} pipeline/01_Write_Region.py -r {rid}'
                )
            else:
                cmd_str = f'uv run python pipeline/01_Write_Region.py -r {rid}'

            batch_command_list.append(cmd_str)

        self.batch_commands = batch_command_list
        return self.batch_commands
