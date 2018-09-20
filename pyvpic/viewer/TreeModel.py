from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt

class TreeItem(object):

    def __init__(self, name, tree, parent=None):
        self.parent_item = parent
        self.name = name
        self.dataset = tree is None

        self.child_items = []
        if not self.dataset:
            if 'groups' in tree:
                self.child_items.extend([TreeItem(item, branch, parent=self)
                                         for item, branch in
                                         tree['groups'].items()])
            if 'datasets' in tree:
                self.child_items.extend([TreeItem(name, None, parent=self)
                                         for name in tree['datasets']])

    def absname(self):
        if self.parent_item:
            return f'{self.parent_item.absname()}/{self.name}'
        return self.name

    def parent(self):
        return self.parent_item

    def child(self, row):
        return self.child_items[row]

    def childCount(self):
        return len(self.child_items)

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

    def data(self, column):
        try:
            if column == 0:
                return self.name
            return ""
        except IndexError:
            return None

class TreeModel(QAbstractItemModel):
    def __init__(self, tree, parent=None):
        super().__init__(parent)
        self.setTree(tree)

    def columnCount(self, parent):
        return 1

    def setTree(self, tree):
        self.beginResetModel()
        self.root_item = TreeItem("VPIC Datasets", tree)
        self.endResetModel()

    def data(self, index, role):
        if not index.isValid():
            return None

        item = index.internalPointer()
        if role == Qt.DisplayRole:
            return item.data(index.column())
        elif role == Qt.UserRole and item.dataset:
            return item.absname().split('/', 1)[1]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.root_item.data(section)
        return None

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self.root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        return parent_item.childCount()
