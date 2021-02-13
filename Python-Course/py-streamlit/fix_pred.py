from itertools import groupby

entities =  ["Mr",
 ".",
 "Trump",
 "’",
 "s",
 "tweets ",
 "began",
 "just ",
 "moments",
 "after",
 "a",
 "Fox",
 "News ",
 "report ",
 "by",
 "Mike",
 "Tobin",
 ",",
 "a",
 "reporter",
 "for",
 "the",
 "network",
 ",",
 "about",
 "protests",
 "in",
 "Minnesota ",
 "and",
 "elsewhere ",
 ".",
 "India",
 "and",
 "China",
 "have ",
 "agreed ",
 "to",
 "peacefully"
 "resolve",
 "a",
 "simmering ",
 "border ",
 "dispute",
 "between",
 "the",
 "world",
 "'",
 "s",
 "two",
 "most",
 "populous",
 "nations",
 ",",
 "officials ",
 "in",
 "New",
 "Delhi",
 "said",
 ".",
  ]


labels =  ["O",
 "O",
 "B-PER",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "B-ORG",
 "B-ORG",
 "O",
 "O",
 "B-PER",
 "B-PER",
 "O",
 "O",
 "O",
 "O",
 "O",
 "B-ORG",
 "O",
 "O",
 "O",
 "O",
 "B-LOC",
 "O",
 "O",
 "O",
 "B-LOC",
 "O",
 "B-LOC",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "O",
 "B-LOC",
 "B-LOC",
 "O",
 "O",
 ]



def format_2(entity_names, entity_types):

    entity_names_2 = []
    entity_types_2 = []
    [(entity_names_2.append(n), entity_types_2.append(t)) for n, t in zip(entity_names, entity_types) if n != "[PAD]"]

    assert len(entity_names_2) == len(entity_types_2)

    # Group same Entity-Types
    groupedLabels = [list(y) for x, y in groupby(entity_types_2)]
    entity_names_2.reverse()

    # Group Entity_Names by Entity-Types
    groupedEntities = []
    for g in groupedLabels:
        groupLocals = []
        for e in g:
            groupLocals.append(entity_names_2.pop())
        groupedEntities.append(groupLocals)
    assert len(groupedEntities) == len(groupedLabels)

    result_dict = {}
    for ent_list, lbl_list in zip(groupedEntities, groupedLabels):
        # print(lbl_list , " >>> " , " ".join(ent_list) )
        if lbl_list[0] not in 'O':
            if lbl_list[0] not in result_dict:
                result_dict[lbl_list[0]] = []

            result_dict[lbl_list[0]].append(" ".join(ent_list))
    # print(result_dict)
    return(result_dict)


def format_1(entity_names, entity_types):
    # Remove the [PAD]s'
    entity_names_2 = []
    entity_types_2 = []
    [(entity_names_2.append(n), entity_types_2.append(t)) for n, t in zip(entity_names, entity_types) if n != "[PAD]"]
    assert len(entity_names_2) == len(entity_types_2)

    # Group same Entity-Types
    groupedEntity = [list(y) for x, y in groupby(entity_types_2)]
    entity_names_2.reverse()

    # Group Entity_Names by Entity-Types
    groupedNames = []
    for g in groupedEntity:
        groupLocals = []
        for e in g:
            groupLocals.append(entity_names_2.pop())
        groupedNames.append(groupLocals)
    assert len(groupedNames) == len(groupedEntity)

    # Prepare the dictionary
    nameEntityDict = {}
    for nms, ent in zip(groupedNames, groupedEntity):
        if ent[0] != "O":
            # nameEntityDict[tuple(nms)] = ent[0]
            nameEntityDict[", ".join(nms)] = ent[0]

    return nameEntityDict


if __name__ == "__main__":
    for e, l in zip(entities, labels):
        print("{:5}  {}".format(l, e))

    print(format_1(entities, labels))
    print(format_2(entities, labels))
