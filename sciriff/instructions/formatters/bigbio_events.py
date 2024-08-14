from sciriff.lib import util


class Formatter:
    """
    Formatter for event extraction from BigBIO datasets.

    :param event_types_map: If `None`, assumes there is only a single event type, which
        will not be output in the formatted data. Otherwise, a string mapping event
        types in the input data to preferred output names, like
        `input1:output1,input2:output2,...`.
    :param role_types_map: A string mapping argument roles in the data to preferred
        output names. Formatted the same as `event_types_map`.
    :param min_arguments: The minimum number of arguments an event must have in order to
        be included; otherwise it will be skipped.
    """

    def __init__(self, event_types_map, role_types_map, min_arguments=1):
        if event_types_map is None:
            self.event_types_map = None
        else:
            self.event_types_map = dict(
                [x.split(":") for x in event_types_map.split(",")]
            )

        self.role_types_map = dict([x.split(":") for x in role_types_map.split(",")])
        self.min_arguments = min_arguments
        self.role_types = list(set(self.role_types_map.values()))

    def _format_event(self, event, entity_dict):
        # In the materials_syntheses dataset, there's only one trigger text per event;
        # should check if this changes for other datasets.
        if len(event["trigger"]["text"]) > 1:
            raise ValueError("Haven't seen this before; investigate.")

        event_args = set()
        for this_arg in event["arguments"]:
            arg_text = entity_dict[this_arg["ref_id"]]["text"]

            if len(arg_text) > 1:
                raise ValueError("Haven't seen this before; investigate.")

            to_add = (arg_text[0], self.role_types_map[this_arg["role"]])
            # Add to set (rather than append to list) to remove duplicates.
            event_args.add(to_add)

        event_args = [list(x) for x in event_args]
        if len(event_args) < self.min_arguments:
            return None

        # Very rarely (twice in the train set), the same entity serves two roles. Let's
        # just skip these to avoid confusion.
        ents = [x[0] for x in event_args]
        if len(ents) != len(set(ents)):
            return None

        res = {
            "trigger": event["trigger"]["text"][0],
            "arguments": event_args,
        }
        if self.event_types_map is not None:
            res["type"] = self.event_types_map[event["type"]]

        return res

    def format_instance(self, instance):
        # Get abstract same way as in `bigbio_ner.py`
        abstract = [" ".join(x["text"]).strip() for x in instance["passages"]]
        abstract = " ".join(abstract)

        entity_dict = util.list_to_dict(instance["entities"], key="id")

        events = []
        for event in instance["events"]:
            formatted_event = self._format_event(event, entity_dict)
            if formatted_event is not None:
                events.append(formatted_event)

        # Return the formatted abstract and the associated events.
        res = {
            "_id": instance["_id"],
            "abstract": abstract,
            "events": events,
            "role_types_list": self.role_types,
        }
        return res

    def format_instances(self, dataset):
        return [self.format_instance(instance) for instance in dataset]
