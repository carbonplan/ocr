from dataclasses import dataclass


@dataclass
class BatchJobs:
    """Dataclass that generates and stores batch commands"""

    region_id: tuple[str, ...]
    run_on_coiled: bool = False

    def generate_batch_commands(self):
        batch_command_list = []
        for rid in self.region_id:
            if self.run_on_coiled:
                cmd_str = f'uv run coiled batch run --name {rid} pipeline/01_Write_Region.py -r {rid} -t project=OCR'
            else:
                cmd_str = f'uv run python pipeline/01_Write_Region.py -r {rid}'

            batch_command_list.append(cmd_str)

        self.batch_commands = batch_command_list
        return self.batch_commands
