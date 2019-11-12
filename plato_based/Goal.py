"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""
import Ontology

__author__ = "Alexandros Papangelis"

import heapq
import math
import os
import pickle
import random

from dialog_action_classes import DialogueActItem, Operator
from DataBase import DataBase, SQLDataBase, JSONDataBase

"""
The Goal represents Simulated Usr goals, that are composed of a set of 
constraints and a set of requests. Goals can be simple or complex, depending 
on whether they have subgoals or not.
"""


class Goal:
    def __init__(self):
        self.constraints = {}  # Dict of <slot, Dialogue Act Item>
        self.requests = {}  # Dict of <slot, Dialogue Act Item>
        self.requests_made = {}  # Dict of <slot, Dialogue Act Item>

        # To be used in the multi-agent setting primarily (where the user does
        # not have access to the ground truth - item in focus - in the
        # dialogue state).
        self.ground_truth = None
        self.subgoals = []
        self.user_model = None

    def __str__(self):
        """
        Generate a string representing the Goal
        :return: a string
        """
        ret = ""

        for c in self.constraints:
            ret += (
                f"\t\tConstr({self.constraints[c].slot}="
                f"{self.constraints[c].value})\n"
            )
        ret += "\t\t-------------\n"
        for r in self.requests:
            ret += f"\t\tReq({self.requests[r].slot})\n"
        ret += "\t\t-------------\n"
        ret += "Sub-goals:\n"
        for sg in self.subgoals:
            for c in sg.constraints:
                if not sg.constraints[c].slot:
                    ret += f"Error! No slot for {c}\n"
                if not sg.constraints[c].value:
                    ret += f"Error! No value for {c}\n"
                ret += (
                    f"\t\tConstr({sg.constraints[c].slot}="
                    f"{sg.constraints[c].value})\n"
                )
            ret += "\t\t--------\n"
        ret += "\n"

        return ret


class GoalGenerator:
    def __init__(self, ontology, database):
        self.ontology = None
        if isinstance(ontology, Ontology.Ontology):
            self.ontology = ontology
        else:
            raise ValueError("Unacceptable ontology type %s " % ontology)

        self.database = None
        if isinstance(database, DataBase):
            self.database = database

        elif isinstance(database, str):
            if database[-3:] == ".db":
                self.database = SQLDataBase(database)
            elif database[-5:] == ".json":
                self.database = JSONDataBase(database)
            else:
                raise ValueError("Unacceptable database type %s " % database)
        else:
            raise ValueError("Unacceptable database type %s " % database)

        self.goals = None

        cursor = self.database.SQL_connection.cursor()

        result = cursor.execute(
            "select * from sqlite_master where type = 'table';"
        ).fetchall()
        if result and result[0] and result[0][1]:
            self.db_table_name = result[0][1]
        else:
            raise ValueError(
                "Goal Generator cannot specify Table Name from "
                "database {0}".format(self.database.db_file_name)
            )

        # Dummy SQL command
        sql_command = "SELECT * FROM " + self.db_table_name + " LIMIT 1;"

        cursor.execute(sql_command)
        self.slot_names = [i[0] for i in cursor.description]

        self.db_row_count = cursor.execute(
            "SELECT COUNT(*) FROM " + self.db_table_name + ";"
        ).fetchall()[0][0]

    def generate(self):

        if self.goals:
            return random.choice(self.goals)

        # Randomly pick an item from the database
        cursor = self.database.SQL_connection.cursor()

        sql_command = (
            "SELECT * FROM "
            + self.db_table_name
            + " WHERE ROWID == ("
            + str(random.randint(1, self.db_row_count))
            + ");"
        )

        cursor.execute(sql_command)
        db_result = cursor.fetchone()

        attempt = 0
        while attempt < 3 and not db_result:
            print(
                "GoalGenerator: Database {0} appears to be empty!".format(self.database)
            )
            print(f"Trying again (attempt {attempt} out of 3)...")

            sql_command = (
                "SELECT * FROM "
                + self.db_table_name
                + " WHERE ROWID == ("
                + str(random.randint(1, self.db_row_count))
                + ");"
            )

            cursor.execute(sql_command)
            db_result = cursor.fetchone()

            attempt += 1

        if not db_result:
            raise LookupError(
                "GoalGenerator: Database {0} appears to be "
                "empty!".format(self.database)
            )

        result = dict(zip(self.slot_names, db_result))

        # Generate goal
        goal = Goal()

        # TODO: Sample from all available operators, not just '='
        # (where applicable)

        inf_slots = random.sample(
            list(self.ontology.ontology["informable"].keys()),
            random.randint(2, len(self.ontology.ontology["informable"])),
        )

        # Sample requests from requestable slots
        req_slots = random.sample(
            self.ontology.ontology["requestable"],
            random.randint(0, len(self.ontology.ontology["requestable"])),
        )

        # Remove slots for which the user places constraints
        # Note: 'name' may or may not be in inf_slots here, and this is
        # randomness is desirable
        for slot in inf_slots:
            if slot in req_slots:
                req_slots.remove(slot)

        # Never ask for specific name unless it is the only constraint
        # if 'name' in inf_slots and len(inf_slots) > 1:
        if "name" in inf_slots:
            inf_slots.remove("name")

        # Shuffle informable and requestable slots to create some variety
        # when pushing into the agenda.
        random.shuffle(inf_slots)
        random.shuffle(req_slots)

        for slot in inf_slots:
            # Check that the slot has a value in the retrieved item
            if slot in result and result[slot]:
                goal.constraints[slot] = DialogueActItem(
                    slot, Operator.EQ, result[slot]
                )

        for slot in req_slots:
            if slot in result:
                goal.requests[slot] = DialogueActItem(slot, Operator.EQ, [])

        return goal

    @staticmethod
    def weighted_random_sample_no_replacement(population, weights, num_samples):
        """
        Samples num_samples from population given the weights

        :param population: a list of things to sample from
        :param weights: weights that bias the sampling from the population
        :param num_samples: how many objects to sample
        :return: a list containing num_samples sampled objects
        """

        elt = [(math.log(random.random()) / weights[i], i) for i in range(len(weights))]
        return [population[i[1]] for i in heapq.nlargest(num_samples, elt)]
